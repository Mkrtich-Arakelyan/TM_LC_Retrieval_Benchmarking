"""Operator-run, one-image / one-frame hardware milestone (python -m entry point)."""

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path

import numpy as np

from hardware.bench_access import (bench_access, BenchBusyError, BenchCancellation,
                                   record_bench_run, require_acquisition_allowed)
from hardware.camera import CameraSettings, CameraTrigger, CeleraCamera
from hardware.generator import GeneratorSettings, LccAcquisitionPlan, RigolDG812
from hardware.laser import LabLaser, LaserSettings
from hardware.slm import MeadowlarkSlm, SlmSettings, SlmReturnCodes


def capture_once(*, slm, camera, laser, output_dir, gray_level, laser_settle_seconds,
                 generator=None, lcc_plan=None):
    """Use supplied adapters; save one constant-gray SLM image and one raw frame.

    The camera must be armed before the image flip. The operator must have
    verified trigger wiring/delay and optical settling. Optional LCC drive
    uses a single amplitude for the capture. Return execution and cleanup results.
    """
    if type(gray_level) is not int or not 0 <= gray_level <= 255:
        raise ValueError("gray_level must be an integer in [0, 255]")
    if (type(laser_settle_seconds) not in (int, float)
            or not math.isfinite(laser_settle_seconds) or laser_settle_seconds < 0):
        raise ValueError("laser_settle_seconds must be finite and nonnegative")
    if (generator is None) != (lcc_plan is None):
        raise ValueError("LCC capture requires both a generator and an explicit acquisition plan")
    with bench_access(acquisition=True):
        return _capture(slm, camera, laser, output_dir, gray_level, laser_settle_seconds,
                        generator, lcc_plan)


def _capture(slm, camera, laser, output_dir, gray_level, laser_settle_seconds, generator, lcc_plan):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)  # before hardware; never overwrite a run
    record_bench_run(output_dir)
    report = {"capture_status": "not_started", "last_stage": "created_output_directory",
              "cleanup": {}, "files": [], "known_image": {"kind": "constant_gray", "value": gray_level}}
    attempted = set()
    try:
        for name, device in (("laser", laser), ("slm", slm)):
            require_acquisition_allowed()
            attempted.add(name)
            report["last_stage"] = f"connecting_{name}"
            device.connect()
            report["last_stage"] = f"connected_{name}"
        image = np.full(slm.image_shape, gray_level, dtype=np.uint8)
        np.save(output_dir / "slm-image.npy", image, allow_pickle=False)
        report["files"].append("slm-image.npy")
        drive = nullcontext()
        if generator is not None:
            require_acquisition_allowed()
            report["last_stage"] = "starting_lcc_drive"
            report["lcc"] = {}
            drive = generator.drive_for_acquisition(lcc_plan, report["lcc"])
        with drive:
            require_acquisition_allowed()
            report["last_stage"] = "starting_laser"
            laser.start_emission()
            report["last_stage"] = "laser_startup_commands_completed"
            if BenchCancellation().wait(laser_settle_seconds):
                require_acquisition_allowed()
            attempted.add("camera")
            report["last_stage"] = "arming_camera"
            camera.connect()
            require_acquisition_allowed()
            report["last_stage"] = "displaying_slm_image"
            slm.display_image(image)
            require_acquisition_allowed()
            report["last_stage"] = "slm_image_displayed"
            frame = camera.capture_frame()
            require_acquisition_allowed()
            report["last_stage"] = "camera_frame_received"
            np.save(output_dir / "camera-frame.npy", frame, allow_pickle=False)
            report["files"].append("camera-frame.npy")
            (output_dir / "raw-frame.bin").write_bytes(frame.tobytes(order="C"))
            report["files"].append("raw-frame.bin")
            report.update(last_stage="raw_frame_saved",
                          frame={"shape": list(frame.shape), "dtype": str(frame.dtype), "order": "C"})
        require_acquisition_allowed()
        report["capture_status"] = "completed"
    except BaseException as exc:
        report.update(capture_status="failed", error=f"{type(exc).__name__}: {exc}",
                      error_notes=getattr(exc, "__notes__", []))
    finally:
        # Stop light first. A failure in any cleanup must not skip another device.
        for name, device in (("laser", laser), ("camera", camera), ("slm", slm)):
            if name not in attempted:
                report["cleanup"][name] = {"status": "not_initialized"}
                continue
            try:
                result = device.disconnect()
                if name == "laser":
                    status = ("UNCONFIRMED" if not result.get("confirmed") else
                              "confirmed" if result.get("serial_close") == "closed" else "failed")
                    report["cleanup"][name] = {"status": status,
                                               "details": result}
                else:
                    report["cleanup"][name] = {"status": "disconnect_returned"}
            except BaseException as exc:
                report["cleanup"][name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        report["laser_transcript"] = laser.transcript
        report["cleanup_ok"] = (report["cleanup"]["laser"]["status"] == "confirmed"
            and all(report["cleanup"][n]["status"] in ("disconnect_returned", "not_initialized")
                    for n in ("camera", "slm")))
        lcc = report.get("lcc")
        if lcc is not None:
            shutdown = lcc.get("shutdown")
            if shutdown is None:
                status = "not_initialized"
            elif not shutdown["confirmed"]:
                status = "UNCONFIRMED"
            elif shutdown["errors"]:
                status = "failed"
            else:
                status = "confirmed"
            report["cleanup"]["generator"] = {"status": status, "details": shutdown}
            report["cleanup_ok"] &= status in ("confirmed", "not_initialized")
        # Console results survive even if saving the diagnostics fails.
        print(json.dumps(report, indent=2))
        (output_dir / "capture-result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_settings(path):
    """Parse/validate everything before creating a device; no hardware calls."""
    return _load_settings(path)[:5]


def _load_settings(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("camera_trigger_confirmed") is not True:
        raise ValueError("Lab camera trigger configuration/wiring must be confirmed before capture")
    slm_data = dict(data["slm"])
    for key in ("dll_path", "lut_path"):
        slm_data[key] = path.parent / slm_data[key]
    slm_data["return_codes"] = SlmReturnCodes(**slm_data["return_codes"])
    slm = SlmSettings(**slm_data)
    if slm.wait_for_trigger or not slm.pulse_on_image_flip or slm.pulse_on_refresh:
        raise ValueError("One-frame wiring requires no SLM input trigger, flip pulse on, refresh pulse off")
    camera_data = dict(data["camera"])
    camera_data["trigger"] = CameraTrigger(**camera_data["trigger"])
    camera = CameraSettings(**camera_data)
    if not camera.trigger.enabled:
        raise ValueError("One-frame capture requires the verified external camera acquisition-start trigger")
    laser = LaserSettings(**data["laser"])
    if not laser.startup_confirmed:
        raise ValueError("Laser startup order must be lab-confirmed before a live test")
    # Validate non-device controls here too, before any serial port/SDK opens.
    gray, settle = data["gray_level"], data["laser_settle_seconds"]
    if type(gray) is not int or not 0 <= gray <= 255:
        raise ValueError("gray_level must be an integer in [0, 255]")
    if type(settle) not in (int, float) or not math.isfinite(settle) or settle < 0:
        raise ValueError("laser_settle_seconds must be finite and nonnegative")
    generator = GeneratorSettings(**data["generator"]) if "generator" in data else None
    lcc_data = data.get("lcc")
    lcc_plan = LccAcquisitionPlan(**lcc_data) if lcc_data is not None else None
    if lcc_plan is not None and (generator is None or not generator.enabled):
        raise ValueError("An LCC acquisition plan requires an enabled generator")
    return slm, camera, laser, gray, settle, generator, lcc_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Lab-verified machine JSON")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--operator-approved", action="store_true", help="Operator authorizes this single live capture")
    args = parser.parse_args(argv)
    if not args.operator_approved:
        parser.error("A live capture requires --operator-approved after lab review")
    try:
        slm_cfg, camera_cfg, laser_cfg, gray, settle, generator_cfg, lcc_plan = _load_settings(args.config)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    try:
        report = capture_once(slm=MeadowlarkSlm(slm_cfg), camera=CeleraCamera(camera_cfg),
                              laser=LabLaser(laser_cfg), output_dir=args.output,
                              gray_level=gray, laser_settle_seconds=settle,
                              generator=RigolDG812(generator_cfg) if lcc_plan is not None else None,
                              lcc_plan=lcc_plan)
    except BenchBusyError as exc:
        parser.exit(1, f"{exc}\n")
    return 0 if report["capture_status"] == "completed" and report["cleanup_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
