import json
import os
import numpy as np
import sys

# Robust base for Windows + Cyrillic path issues
if '__file__' in globals():
    base = os.path.dirname(os.path.abspath(__file__))
else:
    base = os.getcwd()
if not os.path.exists(os.path.join(base, 'patients')):
    # assume we were invoked as python PyCharmMiscProject/diag... from root
    base = os.path.join(os.getcwd(), 'PyCharmMiscProject')
    if not os.path.exists(os.path.join(base, 'patients')):
        base = r'C:\Users\Сомелье по членам\PyCharmMiscProject'

print('Using base dir for diag:', base)
os.chdir(base)

PATIENTS_DIR = 'patients'
patient = 'йцу_йцуц_цццц_213123'
ex_name = 'ПОВОРОТ ГОЛЕНИ'  # user can change this if testing another ex with 3 sessions

# Try to load the real functions if possible
try:
    import rehab_app
    compute_leg_load_moment = rehab_app.compute_leg_load_moment
    load_patient_anthropometrics = rehab_app.load_patient_anthropometrics
    print("Loaded real functions from rehab_app")
except Exception as imp_e:
    print("Could not import full, will use stubs. Error:", imp_e)
    def compute_leg_load_moment(times, angles_by_channel, forces_by_channel, anthro, exercise_name):
        if not times: return None
        n = len(times)
        if not anthro or anthro.get('weight_kg') is None:
            if not forces_by_channel: return None
            proxy = []
            for i in range(n):
                fsum = sum(ch[i] for ch in forces_by_channel if i < len(ch))
                proxy.append(fsum)
            return np.array(proxy)
        M = float(anthro.get('weight_kg') or 70)
        g = 9.81
        Lu = float(anthro.get('upper_link_cm') or 40) / 100
        Lm = float(anthro.get('middle_link_cm') or 40) / 100
        Ll = float(anthro.get('lower_link_cm') or 30) / 100
        m_thigh = 0.10 * M
        m_shank = 0.046 * M
        m_foot = 0.014 * M
        leg_length = Lu + Lm + Ll
        com_factor = 0.55
        loads = []
        for i in range(n):
            if angles_by_channel:
                angle_vals = [ch[i] for ch in angles_by_channel if i < len(ch)]
                avg = np.mean(angle_vals) if angle_vals else 0.0
            else:
                avg = 0.0
            theta = np.radians(avg)
            f_total = 0.0
            for ch in forces_by_channel:
                if i < len(ch):
                    f_total += ch[i]
            force_moment = f_total * (leg_length * 0.45) * abs(np.sin(theta))
            body_support = 0.5 * M * g
            leg_mass_weight = (m_thigh + m_shank + m_foot) * g
            grav_moment = (body_support + leg_mass_weight) * (leg_length * com_factor) * abs(np.sin(theta))
            loads.append(force_moment + grav_moment)
        return np.array(loads)

    def load_patient_anthropometrics(patient_name):
        patient_dir = os.path.join(PATIENTS_DIR, patient_name)
        info_path = os.path.join(patient_dir, 'info.txt')
        data = {'weight_kg': None, 'upper_link_cm': None, 'middle_link_cm': None, 'lower_link_cm': None}
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                data['weight_kg'] = float(info.get('weight_kg')) if info.get('weight_kg') else None
                data['upper_link_cm'] = float(info.get('upper_link_cm')) if info.get('upper_link_cm') else 40
                data['middle_link_cm'] = float(info.get('middle_link_cm')) if info.get('middle_link_cm') else 40
                data['lower_link_cm'] = float(info.get('lower_link_cm')) if info.get('lower_link_cm') else 30
            except Exception as e:
                print("info load err", e)
        return data

patient_dir = os.path.join(PATIENTS_DIR, patient)
anthro = load_patient_anthropometrics(patient)
print("Anthro:", anthro)

ui_folders = []
for f in os.listdir(patient_dir):
    if f.startswith(ex_name + "_"):
        fpath = os.path.join(patient_dir, f)
        if os.path.isdir(fpath) and os.path.exists(os.path.join(fpath, 'angles.png')):
            ui_folders.append(f)
print(f"UI folders for {ex_name}: {len(ui_folders)}")

sessions_data = []
for folder in ui_folders:
    raw_path = os.path.join(patient_dir, folder, 'raw_measurements.json')
    print(f"\n=== Processing {folder} ===")
    print(f"  raw exists: {os.path.exists(raw_path)}")
    if not os.path.exists(raw_path):
        print("  SKIP: no raw")
        continue
    try:
        with open(raw_path, 'r', encoding='utf-8') as fh:
            raw = json.load(fh)
        times_str = raw.get('times', [])
        angles_rows = raw.get('angles', []) or []
        forces_rows = raw.get('forces', []) or []
        print(f"  times: {len(times_str)}")
        print(f"  angles_rows len: {len(angles_rows)}, first row type/len: {type(angles_rows[0]) if angles_rows else None}, {len(angles_rows[0]) if angles_rows and hasattr(angles_rows[0],'__len__') else 'N/A'}")
        print(f"  forces_rows len: {len(forces_rows)}, first row type/len: {type(forces_rows[0]) if forces_rows else None}, {len(forces_rows[0]) if forces_rows and hasattr(forces_rows[0],'__len__') else 'N/A'}")

        T = len(times_str)
        n_a = len(angles_rows[0]) if angles_rows and angles_rows[0] else 0
        n_f = len(forces_rows[0]) if forces_rows and forces_rows[0] else 0
        print(f"  n_a={n_a}, n_f={n_f}")

        angles_by_ch = [[] for _ in range(n_a)] if n_a > 0 else []
        bad_float = False
        for row in angles_rows:
            for ch in range(n_a):
                try:
                    val = float(row[ch]) if row[ch] is not None else 0.0
                    angles_by_ch[ch].append(val)
                except Exception as fe:
                    print(f"    FLOAT ERROR in angles row: {fe}")
                    bad_float = True
                    break
            if bad_float: break

        forces_by_ch = [[] for _ in range(n_f)] if n_f > 0 else []
        for row in forces_rows:
            for ch in range(n_f):
                try:
                    val = float(row[ch]) if row[ch] is not None else 0.0
                    forces_by_ch[ch].append(val)
                except Exception as fe:
                    print(f"    FLOAT ERROR in forces row: {fe}")
                    bad_float = True
                    break
            if bad_float: break

        if bad_float:
            print("  SKIP due to float errors in raw data")
            continue

        forces_N = [[fv * 9.81 / 1000.0 for fv in ch] for ch in forces_by_ch] if forces_by_ch else []
        print(f"  built angles_by_ch lens: {[len(x) for x in angles_by_ch]}")
        print(f"  built forces_N lens: {[len(x) for x in forces_N]}")

        M = None
        try:
            M = compute_leg_load_moment(times_str, angles_by_ch, forces_N, anthro, ex_name) if (angles_by_ch or forces_N) else None
            print(f"  compute returned M len: {len(M) if M is not None else 0}")
        except Exception as ce:
            print(f"  compute EXCEPTION: {ce}")
            M = None

        if M is not None and len(M) > 0:
            peak = float(np.max(M))
            impulse = float(np.trapz(M))
            cv = float(np.std(M) / (np.mean(np.abs(M)) + 1e-9))
            print(f"  GOOD M from compute: len={len(M)}, peak={peak:.1f}")
        else:
            if forces_N:
                TT = len(forces_N[0]) if forces_N and forces_N[0] else 0
                total_f = []
                for t_idx in range(TT):
                    fsum = sum(ch[t_idx] for ch in forces_N if t_idx < len(ch))
                    total_f.append(fsum)
                peak = float(max(total_f)) if total_f else 0.0
                impulse = float(sum(total_f))
                cv = float(np.std(total_f) / (np.mean(np.abs(total_f)) + 1e-9)) if total_f else 0.0
                M = np.array(total_f) if total_f else np.array([])
                print(f"  fallback total_f M len: {len(M)}")
            else:
                # proxy
                try:
                    proxy = []
                    for row in (forces_rows or []):
                        if isinstance(row, (list, tuple)):
                            s = sum(float(x) * 9.81 / 1000.0 for x in row if x is not None)
                            proxy.append(s)
                    if len(proxy) >= 3:
                        M = np.array(proxy)
                        peak = float(np.max(M))
                        impulse = float(np.trapz(M))
                        cv = float(np.std(M) / (np.mean(np.abs(M)) + 1e-9))
                        print(f"  direct row proxy M len: {len(M)}")
                except Exception as epx:
                    print(f"  proxy err: {epx}")

        if M is None or len(M) < 3:
            print(f"  FINAL: no usable M (len={len(M) if M is not None else 0}) -- would skip")
            continue

        sessions_data.append({'folder': folder, 'Mlen': len(M)})
        print(f"  APPENDED with M len {len(M)}")
    except Exception as e:
        print(f"  BIG EXCEPTION for folder: {e}")
        import traceback
        traceback.print_exc()

print(f"\n=== RESULT: collected {len(sessions_data)} valid ===")
for s in sessions_data:
    print(s)
