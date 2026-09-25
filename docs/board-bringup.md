# Coralboard bring-up

A checklist from unboxing to running our memory step on the NPU. It was compiled from Google's and Synaptics' docs and the Torq compiler source on 2026-09-24, and **hasn't been tried on our board yet**. Update it as we go.

## What you need

- A **USB-C data cable** (not charge-only) from the PC to the board. The board has one USB-C port for both 5 V power and data. It's 5 V only, with no USB Power Delivery negotiation. ([user guide](https://developers.google.com/coral/products/SL2610-user-guide))
- **No SD card.** The board boots from built-in eMMC, and SD boot is "not supported out-of-the-box" on the Limited Edition. ([get started](https://developers.google.com/coral/products/SL2610-get-started))
- A #0 Phillips screwdriver, only if you mount the sensor HAT.
- Optional, for a serial console: the sensor HAT's USB-C port plus the [Silicon Labs CP210x driver](https://www.silabs.com/developer-tools/usb-to-uart-bridge-vcp-drivers), at 115200 baud.

## 1. First shell (preloaded image)

1. Install [Android SDK Platform-Tools](https://developer.android.com/tools/releases/platform-tools) on the PC.
2. Plug USB-C into the **board** (not the HAT), then run `adb shell`. You're `root` with no password.
3. Record the image version: `cat /etc/os-release`. We want Astra SDK **scarthgap_6.12_v2.5.0** or newer.

## 2. Network (the board has no Ethernet)

Share the PC's internet over USB, per the [user guide](https://developers.google.com/coral/products/SL2610-user-guide):

1. Open Windows Network Connections, right-click your internet adapter, choose Properties → Sharing, and share to **"UsbNcm Host Device"**.
2. On the board, run `udhcpc -i usb0`, then `ip addr show usb0` to get its IP.
3. From the PC, `ssh root@<board-ip>`.

Windows sharing doesn't survive a PC reboot, so repeat step 1 after restarting. Wi-Fi needs an optional M.2 module (AMPAK AP12611_M2) that isn't in the box.

## 3. Reflash to SDK 2.5.0, if the image is older

1. Download the image with [`get_sl2619_coralboard_oobe_scarthgap_6.12_v2.5.0_image.bat`](https://github.com/synaptics-astra/sdk/releases/download/scarthgap_6.12_v2.5.0/get_sl2619_coralboard_oobe_scarthgap_6.12_v2.5.0_image.bat).
2. Get the flasher from [synaptics-astra/usb-tool](https://github.com/synaptics-astra/usb-tool): `bin/win/astra-update.exe`, the `astra-usbboot-images/sl2610_coralboard` boot images, and `update_emmc.bat`. Put the image folder, named `eMMCimg`, next to it.
3. Install the [Google USB Driver](https://developer.android.com/studio/run/win-usb). Since SDK 2.4 the board flashes over fastboot.
4. Enter USB download mode: **hold USER and click RESET**. Then run `update_emmc.bat`.
5. **Let the board reset itself when it's done.** Resetting it by hand can corrupt partitions.
6. Keep work in `/home/root`, because the root partition is small.

## 4. Compiler on the PC (WSL)

The Torq compiler only runs on x86-64 Linux. On Windows, that means WSL2.

```powershell
wsl --install -d Ubuntu-24.04      # 24.04 specifically: the wheels need glibc >= 2.39
```

Then, inside WSL:

```bash
python3.12 -m venv ~/torq && source ~/torq/bin/activate
pip install torq-compiler torq-runtime        # v2.2.1 as of 2026-09-23
```

Match the versions: the compiler, the board's `torq-runtime`, and the image's kernel module must agree, or older runtimes reject newer models with "bytecode version mismatch". On the board, run `pip install torq-runtime` (the board has Python 3.12), pinned to the compiler's version.

## 5. First test: the delta-rule step in bf16

bf16 is Torq's main path, and every op the step needs is listed as NPU-supported in bf16. Only try int8 after that works.

1. Write the step `(k, v, S) -> (readout, S_new)` in PyTorch using explicit `matmul` and broadcast multiply (no einsum), with `lam` and `beta` as constants. Do **argmax on the host**, since NPU argmax is int8-only.
2. Export with `torch.onnx.export(..., dynamo=True)`.
3. Compile with the CPU fallbacks turned off, which proves every op landed on the NPU:
   ```bash
   torq-compile step.onnx -o step.vmfb --torq-convert-dtypes --torq-convert-io-dtype --torq-disable-host --torq-disable-css
   ```
4. Check it on the host simulator (`torq-lab verify`), then run it on the board with the Python API, feeding `S_new` back as `S` on the device (`device_outputs=True`, as documented for KV caches):
   ```python
   from torq.runtime import VMFBInferenceRunner
   r = VMFBInferenceRunner("step.vmfb", device_uri="torq")
   ```
5. Compare against our simulator on the same stream. That comparison is the result.

## Known risks

- **No int4 state.** Torq supports int4 only for *weights*, so our INT4 results can't be reproduced on this chip. The state has to be int8, int16, or bf16.
- **Open int8 bug**, [torq-compiler#28](https://github.com/synaptics-torq/torq-compiler/issues/28): adding two int8 tensors with different scales can compile to a malformed add. `lam*S + outer(...)` is exactly that pattern. This is why bf16 comes first.
- **int8 scales for fed-back state aren't pinned automatically.** ONNX Runtime calibrates `S` and `S_new` independently, so we'd have to force identical scales (`TensorQuantOverrides`) or hand-write the graph.
- **int8 matrix-vector products accumulate in int16**, which can overflow at d=128. Use `matmul` (int32 accumulate).
- **Rounding:** the NPU requantizes with round-to-nearest, ties toward +∞ ("NTP"). That differs from our simulator's round-half-to-even only at exact ties.
- **Speed isn't the point.** A single step is tiny, and per-call overhead will dominate. We're testing numerics, not throughput.

## Sources

Google: [get started](https://developers.google.com/coral/products/SL2610-get-started), [user guide](https://developers.google.com/coral/products/SL2610-user-guide). Synaptics: [Astra SDK 2.5.0](https://github.com/synaptics-astra/sdk/releases/tag/scarthgap_6.12_v2.5.0), [Torq manual](https://synaptics-torq.github.io/torq-compiler/v/latest/user-manual/introduction.html) ([ops](https://synaptics-torq.github.io/torq-compiler/v/latest/user-manual/ops.html), [runtime](https://synaptics-torq.github.io/torq-compiler/v/latest/user-manual/torq_runtime.html), [CPU fallback](https://synaptics-torq.github.io/torq-compiler/v/latest/user-manual/css_host_fallback.html)), [Bring Your Own Model](https://developer.synaptics.com/docs/sl/sl2600/tutorials/bring-your-own-model), [sl2610-examples](https://github.com/synaptics-astra-demos/sl2610-examples).
