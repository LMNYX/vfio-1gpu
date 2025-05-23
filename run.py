import os, subprocess
import re
import time
import logging
import inquirer
import shutil
import string
import xml.etree.ElementTree as ET
from pathlib import Path


logger = logging.getLogger(__name__)
logging.basicConfig(filename='1gpuvfio.log', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

def get_display_manager():
    result = subprocess.run(
        ["grep", "ExecStart=", "/etc/systemd/system/display-manager.service"],
        capture_output=True,
        text=True
    )

    if (result.stdout == ""):
        raise ValueError("/etc/systemd/system/display-manager.service is empty.")

    match = re.search(r'ExecStart=.*/([^/\s]+)', result.stdout)

    if not match: raise ValueError("/etc/systemd/system/display-manager.service is empty.")

    return match.group(1)

def get_cpu_vendor():
    result = subprocess.run(
    ["lscpu"],
    capture_output=True,
    text=True
)

    for line in result.stdout.splitlines():
        if "Vendor ID" in line:
            return "AMD" if line.split(":")[1].strip() == "AuthenticAMD" else "Intel"

def get_grub_cmdline_default():
    with open("/etc/default/grub") as file:
        for line in file:
            if re.search(r'^GRUB_CMDLINE_LINUX_DEFAULT=".*"', line):
                return line.strip()

def add_iommu_to_grub(vendor: str, grub_config: str):

    pattern = r'^(GRUB_CMDLINE_LINUX_DEFAULT=")(.*?)(")$'

    def replacer(match):
        before, args, after = match.groups()
        if f"{vendor.lower()}_iommu=on" not in args:
            args = args.strip() + f" {vendor.lower()}_iommu=on"
        return f"{before}{args}{after}"

    return re.sub(pattern, replacer, grub_config, flags=re.MULTILINE)

def modify_file(file_path: str, old: str, new: str):
    cmd = [
        "sudo", "sed", "-i",
        f"s|{old}|{new}|g",
        file_path
    ]
    subprocess.run(cmd)

def get_iommu_devices():
    iommu_groups_path = "/sys/kernel/iommu_groups"
    pci_pattern = re.compile(r"^[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]")  # pattern like 2a:00.0

    groups = [d for d in os.listdir(iommu_groups_path) if os.path.isdir(os.path.join(iommu_groups_path, d))]
    try:
        groups.sort(key=lambda x: int(x))
    except ValueError:
        groups.sort()

    matched_lines = []

    for group in groups:
        devices_path = os.path.join(iommu_groups_path, group, "devices")
        if os.path.exists(devices_path):
            devices = os.listdir(devices_path)
            for device in devices:
                result = subprocess.run(
                    ["lspci", "-nns", device],
                    capture_output=True,
                    text=True,
                    check=False
                )
                output = result.stdout.strip()
                if output and pci_pattern.match(output):
                    matched_lines.append(output)

    return matched_lines

def filter_pci_devices(devices):

    pattern = re.compile(r'\[(03|038)00\]')
    
    filtered = [device for device in devices if pattern.search(device)]
    return filtered

def find_gpu_rom(identifier: str) -> str | None:
    try:
        result = subprocess.run(
            ['find', '/sys/devices', '-name', 'rom'],
            capture_output=True,
            text=True,
            check=True
        )

        paths = result.stdout.strip().split('\n')
        
        for path in paths:
            if identifier in path:
                return path
                
        return None  # Not found
    
    except subprocess.CalledProcessError as e:
        raise ValueError("GPU ROM not found.")

def dump_gpu_rom(rom_path: str, output_path: str) -> bool:
    try:

        enable_cmd = f"echo 1 | sudo tee {rom_path}"
        subprocess.run(enable_cmd, shell=True, check=True)

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as out_file:
            subprocess.run(['sudo', 'cat', rom_path], stdout=out_file, check=True)

        return True

    except subprocess.CalledProcessError as e:
        print(f"Error during ROM dump: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def get_gpu_vendor():
    try:
        output = subprocess.check_output(['lspci'], text=True)
        
        gpu_lines = [line for line in output.splitlines() if 'VGA' in line or '3D controller' in line or 'Display controller' in line]

        vendors = []
        for line in gpu_lines:
            parts = line.split(':', 2)
            if len(parts) >= 3:
                vendor_info = parts[2].strip()
                vendor_name = vendor_info.split()[0]
                vendors.append(vendor_name)

        return vendors if vendors else ["Unknown"]

    except FileNotFoundError:
        return ["Error: 'lspci' command not found. Install 'pciutils'."]
    except subprocess.CalledProcessError as e:
        return [f"Error executing lspci: {e}"]

def find_starting_with(strings, prefix):
    for s in strings:
        if s.startswith(prefix):
            return s
    return None

def is_ascii_printable(byte):
    return 32 <= byte <= 126

def find_trim_point(data, keyword=b'VIDEO', min_ascii_len=6):
    keyword_index = data.find(keyword)
    if keyword_index == -1:
        raise ValueError("Keyword 'VIDEO' not found in ROM")

    i = keyword_index
    while i > 0:
        if is_ascii_printable(data[i - 1]):
            i -= 1
        else:
            break

    start = i
    while start > 0:
        if is_ascii_printable(data[start - 1]):
            start -= 1
        else:
            break

    if keyword_index - start < min_ascii_len:
        raise ValueError("No clear ASCII signature found before 'VIDEO'")

    return start

def patch_rom(input_path, output_path=None):
    with open(input_path, 'rb') as f:
        rom_data = f.read()

    try:
        trim_point = find_trim_point(rom_data)
    except ValueError as e:
        print(f"Error: {e}")
        return False

    patched_data = rom_data[trim_point:]

    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_patched{ext}"

    with open(output_path, 'wb') as f:
        f.write(patched_data)

    print(f"Patched ROM saved to {output_path}")
    return True

def list_vtconsole_files():
    try:
        # Use `ls -p` and filter out directories (they end with /)
        result = subprocess.run(
            ['ls', '-p', '/sys/class/vtconsole/'],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Filter out entries that end with '/' (directories)
        files = [f"/sys/class/vtconsole/{line}" for line in result.stdout.splitlines() if not line.endswith('/')]
        return files
    except subprocess.CalledProcessError as e:
        print("Error listing files:", e.stderr)
        return []


def get_qemu_machines():
    try:
        result = subprocess.run(
            ['find', '/etc/libvirt/qemu', '-maxdepth', '1', '-type', 'f', '-name', '*.xml'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        xml_files = result.stdout.strip().split('\n') if result.stdout else []
        return xml_files
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")
        return []

def add_pcie_device_to_vm(vm_name, pcie_address, rom_file="/home/test", guest_domain="0x0000", guest_bus="0x06", guest_slot="0x00", guest_function="0x0"):
    domain = libvirt.open(None).lookupByName(vm_name)
    xml = domain.XMLDesc()
    tree = ET.fromstring(xml)

    bus, rest = pcie_address.split(":")
    slot, function = rest.split(".")

    hostdev = ET.Element("hostdev", mode="subsystem", type="pci", managed="yes")
    source = ET.SubElement(hostdev, "source")
    address = ET.SubElement(source, "address", domain="0x0000", bus=f"0x{bus}", slot=f"0x{slot}", function=f"0x{function}")

    ET.SubElement(hostdev, "rom", file=rom_file)
    ET.SubElement(hostdev, "address", type="pci", domain=guest_domain, bus=guest_bus, slot=guest_slot, function=guest_function)

    devices = tree.find("devices")
    devices.append(hostdev)

    new_xml = ET.tostring(tree).decode()
    conn = libvirt.open(None)
    conn.defineXML(new_xml)
    print(f"PCIe device {pcie_address} added to VM '{vm_name}'.")

def main():
    display_manager = get_display_manager()
    logger.info(f"Display manager is {display_manager}")
    cpu_vendor = get_cpu_vendor()
    gpu_vendor = get_gpu_vendor()[0]
    logger.info(f"CPU vendor is {cpu_vendor}")
    logger.info(f"GPU vendor is {gpu_vendor}")

    home_dir = os.path.expanduser("~")

    logger.info(f"User home directory is {home_dir}")

    stamp = time.time()
    backup_folder = f"{home_dir}/1gpuvfio_backup.{stamp}"
    logger.info(f"Creating a directory for backups at {backup_folder}")
    try:
        os.mkdir(f"{backup_folder}")
    except OSError:
        logger.info(f"Tried creating a backup directory ({backup_folder}), but it already exists...")

    logger.info("Copying grub configuration to the backup folder.")
    l = subprocess.run(["cp", "/etc/default/grub", f"{backup_folder}/grub.bkp"])

    logger.info("Checking for iommu in grub setup...")
    grub_cmdline = get_grub_cmdline_default()
    iommu_enabled = f"{cpu_vendor.lower()}_iommu=on" in grub_cmdline
    logger.info(f"iommu enabled for {cpu_vendor} ... {iommu_enabled}")
    
    if not iommu_enabled:
        new_grub_cmdline = add_iommu_to_grub(cpu_vendor, grub_cmdline)
        logger.info(f"Modified cmdline is {new_grub_cmdline}")

        modify_file('/etc/default/grub', grub_cmdline, new_grub_cmdline)

        subprocess.run(["sudo", "grub-mkconfig", "-o", "/boot/grub/grub.cfg"])

        logger.info("IOMMU enabled, you have to reboot and run the script again.")

    logger.info("Checking IOMMU groups...")
    iommu_groups = get_iommu_devices()

    questions = [
    inquirer.List('gpu_device',
                    message="What device is your GPU?",
                    choices=filter_pci_devices(iommu_groups) + ["Not in the list"],
                ),
    ]
    answers = inquirer.prompt(questions)
    
    if(answers['gpu_device'] == "Not in the list"):
        questions = [
        inquirer.List('gpu_device',
                        message="What device is your GPU?",
                        choices=iommu_groups + ["Not in the list"],
                    ),
        ]
        answers = inquirer.prompt(questions)

        if(answers['gpu_device'] == "Not in the list"):
            raise ValueError("GPU is not present in IOMMU devices.")

    logger.info(f"GPU device selected is {answers['gpu_device']}")
    gpu_iommu = answers['gpu_device']
    gpu_iommu_n = re.match(r'^\w+\:\w+\.\w+', gpu_iommu)[0]

    manual_intervention_rom = False

    try:
        gpu_rom_path = find_gpu_rom(gpu_iommu_n)
        logger.info(f"GPU ROM path is {gpu_rom_path}")
        logger.info(f"Dumping GPU rom to {backup_folder}/gpu.rom")
        dump_gpu_rom(gpu_rom_path, f"{backup_folder}/gpu.rom")
        logger.info(f"Making a GPU rom backup to {backup_folder}/gpu.rom.bkp")
        shutil.copyfile(f"{backup_folder}/gpu.rom", f"{backup_folder}/gpu.rom.bkp")
    except ValueError:
        manual_intervention_rom = True
        logger.warn("GPU rom not found locally. Most likely your GPU is old, protected or plain stupid.")
        logger.warn("Please find your GPUs rom somewhere else, for example: https://www.techpowerup.com/vgabios/")
        logger.warn(f"After you got your ROM please put it in {backup_folder} and name it external_gpu.rom")
        logger.warn("Press ENTER to continue. (PRESS AFTER ROM IS PLACED IN THE DIRECTORY)")
        input()
        shutil.copyfile(f"{backup_folder}/external_gpu.rom", f"{backup_folder}/gpu.rom")
        shutil.copyfile(f"{backup_folder}/external_gpu.rom", f"{backup_folder}/gpu.rom.bkp")
        os.remove(f"{backup_folder}/external_gpu.rom")

    logger.info("GPU rom and its backup made.")

    logger.info("Finding GPU audio interface...")
    gpu_audio_iommu = find_starting_with(iommu_groups, gpu_iommu_n[:-1] + "1")
    gpu_audio_iommu_n = re.match(r'^\w+\:\w+\.\w+', gpu_audio_iommu)[0]
    logger.info(f"Found: {gpu_audio_iommu}")

    questions = [
        inquirer.List('gpu_audio_device',
                        message="What's your GPU High Definition audio device?",
                        choices=[gpu_audio_iommu] + ["Not in the list"],
                    ),
        ]
    answers = inquirer.prompt(questions)

    if (answers["gpu_audio_device"] == "Not in the list"):
        raise ValueError("GPU and GPU's High Definition audio is not in the same group. Manual intervention required. Please isolate them.")

    logger.info("Proceeding to patch the GPU rom...")
    attempt_patch = patch_rom(f"{backup_folder}/gpu.rom", f"{backup_folder}/gpu_patched.rom")
    if (not attempt_patch):
        logging.warn("Patching was not successful.")
        if(manual_intervention_rom):
            logger.warn("ROM was provided manually. Please inspect the ROM with the HEX editor and confirm it's already trimmed.")
            logger.info("If the ROM isn't trimmed manually, close this script and manually trim the ROM and run the script one more time.")
            logger.info("If the ROM is trimmed correctly press ENTER to continue.")
            input()
        shutil.copyfile(f"{backup_folder}/gpu.rom",f"{backup_folder}/gpu_patched.rom")

    logger.info("!! Initial setup complete.")
    logger.info("This script assumes you already have a QEMU virtual machine created in advance.")
    qemu_machines = get_qemu_machines()

    questions = [
        inquirer.List('qemu_machine',
                        message="Select a QEMU machine we are modifying",
                        choices=qemu_machines,
                    ),
        ]
    answers = inquirer.prompt(questions)

    machine_path = answers['qemu_machine']
    machine_name = Path(machine_path).stem

    logger.info(f"QEMU VM is {machine_name} ({machine_path})")

    logger.info("Preparing for hooks to be created")

    subprocess.run(["sudo", "mkdir", "/etc/libvirt/hooks"])
    subprocess.run(["sudo", "mkdir", "/etc/libvirt/hooks/qemu.d"])
    subprocess.run(["sudo", "mkdir", f"/etc/libvirt/hooks/qemu.d/{machine_name}"])
    subprocess.run(["sudo", "mkdir", f"/etc/libvirt/hooks/qemu.d/{machine_name}/prepare"])
    subprocess.run(["sudo", "mkdir", f"/etc/libvirt/hooks/qemu.d/{machine_name}/prepare/begin"])
    subprocess.run(["sudo", "mkdir", f"/etc/libvirt/hooks/qemu.d/{machine_name}/release"])
    subprocess.run(["sudo", "mkdir", f"/etc/libvirt/hooks/qemu.d/{machine_name}/release/end"])
    subprocess.run(["sudo", "wget", "https://raw.githubusercontent.com/PassthroughPOST/VFIO-Tools/refs/heads/master/libvirt_hooks/qemu", "-O", "/etc/libvirt/hooks/qemu"])
    subprocess.run(["sudo", "chmod", "+x", "/etc/libvirt/hooks/qemu"])

    kvm_conf_file_contents = f"""VIRSH_GPU_VIDEO=pci_0000_{gpu_iommu_n.replace(":", "_")}
VIRSH_GPU_AUDIO=pci_0000_{gpu_audio_iommu_n.replace(":", "_")}"""
    logger.info(kvm_conf_file_contents)
    with open('_', 'w') as f:
        f.write(kvm_conf_file_contents)
    
    subprocess.run(["sudo", "mv", "_", "/etc/libvirt/hooks/kvm.conf"])

    prepare_begin_script_contents = f"""set -x
source "/etc/libvirt/hooks/kvm.conf"

# Stop display manager
systemctl stop {display_manager}.service

# Unbind VTConsoles"""

    for vtcon in list_vtconsole_files():
        prepare_begin_script_contents += f"echo 0 > {vtcon}"

    prepare_begin_script_contents += """

# Unbind EFI-framebuffer
echo efi-framebuffer.0 > /sys/bus/platform/drivers/efi-framebuffer/unbind

sleep 10"""

    if (gpu_vendor == "NVIDIA"):
        prepare_begin_script_contents += """
# Unload GPU NVIDIA
modprobe -r nvidia_drm
modprobe -r nvidia_modeset
modprobe -r drm_kms_helper
modprobe -r nvidia
modprobe -r i2c_nvidia_gpu
modprobe -r drm
modprobe -r nvidia_uvm
"""
    elif (gpu_vendor == "Advanced"):
        prepare_begin_script_contents += """
# Unload GPU AMD
modprobe -r snd_hda_codec_hdmi
modprobe -r snd_hda_intel
modprobe -r amdgpu
modprobe -r drm_kms_helper
modprobe -r drm

"""
    else:
        logging.log("GPU is probably Intel. Modify hooks manually as it's unsupported.")
    prepare_begin_script_contents += """
# Unbind GPU
virsh nodedev-detach $VIRSH_GPU_VIDEO
virsh nodedev-detach $VIRSH_GPU_AUDIO

# Load vfio
modprobe vfio
modprobe vfio_pci
modprobe vfio_iommu_type1"""

    release_revert_script_contents = f"""set -x
source "/etc/libvirt/hooks/kvm.conf"
# unload vfio
modprobe -r vfio_pci
modprobe -r vfio_iommu
modprobe -r vfio

# Rebind GPU
virsh nodedev-reattach $VIRSH_GPU_VIDEO
virsh nodedev-reattach $VIRSH_GPU_AUDIO

# rebind VTConsoles
"""
    for vtcon in list_vtconsole_files():
        release_revert_script_contents += f"echo 1 > {vtcon}\n"
    if (gpu_vendor == "NVIDIA"):
        release_revert_script_contents += """

nvidia-xconfig --query-gpu-info > /dev/null 2>&1
"""

    release_revert_script_contents += """
echo "efi-framebuffer.0" > /sys/bus/platform/drivers/efi-framebuffer/bind
"""

    if (gpu_vendor == "NVIDIA"):
        release_revert_script_contents += """
# Load GPU NVIDIA
modprobe nvidia_drm
modprobe nvidia_modeset
modprobe drm_kms_helper
modprobe nvidia
modprobe i2c_nvidia_gpu
modprobe drm
modprobe nvidia_uvm
"""
    elif (gpu_vendor == "Advanced"):
        release_revert_script_contents += """
# Load GPU AMD
modprobe snd_hda_codec_hdmi
modprobe snd_hda_intel
modprobe amdgpu
modprobe drm_kms_helper
modprobe drm"""
    else:
        logging.log("GPU is probably Intel. Modify hooks manually as it's unsupported.")
    release_revert_script_contents += f"""

# Start display manager
systemctl start {display_manager}.service
"""
    
    with open('_', 'w') as f:
        f.write(prepare_begin_script_contents)
    
    subprocess.run(["sudo", "mv", "_", f"/etc/libvirt/hooks/qemu.d/{machine_name}/prepare/begin/start.sh"])
    subprocess.run(["sudo", "chmod", "+x", f"/etc/libvirt/hooks/qemu.d/{machine_name}/prepare/begin/start.sh"])

    with open('_', 'w') as f:
        f.write(release_revert_script_contents)
    
    subprocess.run(["sudo", "mv", "_", f"/etc/libvirt/hooks/qemu.d/{machine_name}/release/end/revert.sh"])
    subprocess.run(["sudo", "chmod", "+x", f"/etc/libvirt/hooks/qemu.d/{machine_name}/release/end/revert.sh"])

    logging.log(f"Hooks created at /etc/libvirt/hooks/qemu.d/{machine_name}")

    add_pcie_device_to_vm(machine_name, gpu_iommu_n, f"{backup_folder}/gpu_patched.rom")
    add_pcie_device_to_vm(machine_name, gpu_audio_iommu_n, f"{backup_folder}/gpu_patched.rom")

if __name__ == "__main__":
    logger.info("Starting the process...")
    main()