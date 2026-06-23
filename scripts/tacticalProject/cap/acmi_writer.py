from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AcmiRecorder:
    """Tacview ACMI writer for CAP simulation runs."""

    removed_ids: set[str] = field(default_factory=set)

    def reset(self) -> None:
        self.removed_ids.clear()

    def write_header(self, filepath: str) -> None:
        with open(filepath, 'w', encoding='utf-8-sig') as handle:
            handle.write("FileType=text/acmi/tacview\n")
            handle.write("FileVersion=2.1\n")
            handle.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")

    def write_frame(self, filepath: str, env, timestamp: float) -> None:
        lines = [f"#{timestamp:.2f}\n"]

        for uid, sim in env._jsbsims.items():
            if sim.is_alive:
                lon, lat, alt = sim.get_geodetic()
                roll, pitch, yaw = sim.get_rpy() * 180 / np.pi
                model_name = sim.model.upper()
                if model_name == "SU27SK":
                    model_name = "SU27"
                elif model_name == "F16":
                    model_name = "F16"
                lines.append(
                    f"{uid},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
                    f"Name={model_name},Type=Air+FixedWing,ShortName={uid},Color={sim.color}\n"
                )
            elif uid not in self.removed_ids:
                self.removed_ids.add(uid)

        for sim in env._tempsims.values():
            if hasattr(sim, 'log'):
                log_msg = sim.log()
                if log_msg:
                    lines.append(log_msg + "\n")
            elif hasattr(sim, 'is_alive') and sim.is_alive:
                lon, lat, alt = sim.get_geodetic()
                roll, pitch, yaw = sim.get_rpy() * 180 / np.pi
                missile_name = str(getattr(sim, 'uid', '') or type(sim).__name__).upper()
                lines.append(
                    f"{sim.uid},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
                    f"Name={missile_name},Type=Air+Missile,ShortName={missile_name},Color={sim.color}\n"
                )

        with open(filepath, 'a', encoding='utf-8-sig') as handle:
            handle.writelines(lines)
