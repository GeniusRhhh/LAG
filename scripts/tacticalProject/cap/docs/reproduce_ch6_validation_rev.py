from pathlib import Path
import subprocess, sys
root = Path(__file__).resolve().parent
subprocess.check_call([sys.executable, str(root / 'build_ch6_0511_rev.py')])
