#!/usr/bin/env python3

"""
This example shows usage of Camera Control message as well as ColorCamera configInput to change crop x and y
Uses 'WASD' controls to move the crop window, 'C' to capture a still image, 'T' to trigger autofocus, 'IOKL,.[]'
for manual exposure/focus/white-balance:
  Control:      key[dec/inc]  min..max
  exposure time:     I   O      1..33000 [us]
  sensitivity iso:   K   L    100..1600
  focus:             ,   .      0..255 [far..near]
  white balance:     [   ]   1000..12000 (light color temperature K)
To go back to auto controls:
  'E' - autoexposure
  'F' - autofocus (continuous)
  'B' - auto white-balance

Other controls:
'1' - AWB lock (true / false)
'2' - AE lock (true / false)
'3' - Select control: AWB mode
'4' - Select control: AE compensation
'5' - Select control: anti-banding/flicker mode
'6' - Select control: effect mode
'7' - Select control: brightness
'8' - Select control: contrast
'9' - Select control: saturation
'0' - Select control: sharpness
'[' - Select control: luma denoise
']' - Select control: chroma denoise

For the 'Select control: ...' options, use these keys to modify the value:
  '-' or '_' to decrease
  '+' or '=' to increase
"""

import depthai as dai
import cv2

# Step size ('W','A','S','D' controls)
STEP_SIZE = 8
# Manual exposure/focus/white-balance set step
EXP_STEP = 500  # us
ISO_STEP = 50
LENS_STEP = 3
WB_STEP = 200

# On some host systems it's faster to display uncompressed video
videoMjpeg = False

def clamp(num, v0, v1):
    return max(v0, min(num, v1))

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
videoEncoder = pipeline.create(dai.node.VideoEncoder)
stillEncoder = pipeline.create(dai.node.VideoEncoder)

#camRgb.initialControl.setAutoFocusLensRange(90, 204)

controlIn = pipeline.create(dai.node.XLinkIn)
configIn = pipeline.create(dai.node.XLinkIn)
videoMjpegOut = pipeline.create(dai.node.XLinkOut)
stillMjpegOut = pipeline.create(dai.node.XLinkOut)
previewOut = pipeline.create(dai.node.XLinkOut)

controlIn.setStreamName('control')
configIn.setStreamName('config')
videoMjpegOut.setStreamName('video')
stillMjpegOut.setStreamName('still')
previewOut.setStreamName('preview')

# Properties
camRgb.setVideoSize(640, 360)
camRgb.setPreviewSize(300, 300)
videoEncoder.setDefaultProfilePreset(camRgb.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
stillEncoder.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)

# Linking
camRgb.still.link(stillEncoder.input)
camRgb.preview.link(previewOut.input)
controlIn.out.link(camRgb.inputControl)
configIn.out.link(camRgb.inputConfig)
videoEncoder.bitstream.link(videoMjpegOut.input)
stillEncoder.bitstream.link(stillMjpegOut.input)
if videoMjpeg:
    camRgb.video.link(videoEncoder.input)
    videoEncoder.bitstream.link(videoMjpegOut.input)
else:
    camRgb.video.link(videoMjpegOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Get data queues
    controlQueue = device.getInputQueue('control')
    configQueue = device.getInputQueue('config')
    previewQueue = device.getOutputQueue('preview')
    videoQueue = device.getOutputQueue('video')
    stillQueue = device.getOutputQueue('still')

    # Max cropX & cropY
    maxCropX = (camRgb.getResolutionWidth() - camRgb.getVideoWidth()) / camRgb.getResolutionWidth()
    maxCropY = (camRgb.getResolutionHeight() - camRgb.getVideoHeight()) / camRgb.getResolutionHeight()

    # Default crop
    cropX = 0
    cropY = 0
    sendCamConfig = True

    # Defaults and limits for manual focus/exposure controls
    lensPos = 150
    lensMin = 0
    lensMax = 255

    expTime = 20000
    expMin = 1
    expMax = 33000

    sensIso = 800
    sensMin = 100
    sensMax = 1600
    
    wbManual = 4000
    wbMin = 1000
    wbMax = 12000

    # TODO make auto-iterable
    awb_mode_idx = -1
    awb_mode_list = [
        dai.CameraControl.AutoWhiteBalanceMode.OFF,
        dai.CameraControl.AutoWhiteBalanceMode.AUTO,
        dai.CameraControl.AutoWhiteBalanceMode.INCANDESCENT,
        dai.CameraControl.AutoWhiteBalanceMode.FLUORESCENT,
        dai.CameraControl.AutoWhiteBalanceMode.WARM_FLUORESCENT,
        dai.CameraControl.AutoWhiteBalanceMode.DAYLIGHT,
        dai.CameraControl.AutoWhiteBalanceMode.CLOUDY_DAYLIGHT,
        dai.CameraControl.AutoWhiteBalanceMode.TWILIGHT,
        dai.CameraControl.AutoWhiteBalanceMode.SHADE,
    ]

    anti_banding_mode_idx = -1
    anti_banding_mode_list = [
        dai.CameraControl.AntiBandingMode.OFF,
        dai.CameraControl.AntiBandingMode.MAINS_50_HZ,
        dai.CameraControl.AntiBandingMode.MAINS_60_HZ,
        dai.CameraControl.AntiBandingMode.AUTO,
    ]

    effect_mode_idx = -1
    effect_mode_list = [
        dai.CameraControl.EffectMode.OFF,
        dai.CameraControl.EffectMode.MONO,
        dai.CameraControl.EffectMode.NEGATIVE,
        dai.CameraControl.EffectMode.SOLARIZE,
        dai.CameraControl.EffectMode.SEPIA,
        dai.CameraControl.EffectMode.POSTERIZE,
        dai.CameraControl.EffectMode.WHITEBOARD,
        dai.CameraControl.EffectMode.BLACKBOARD,
        dai.CameraControl.EffectMode.AQUA,
    ]

    ae_comp = 0
    ae_lock = False
    awb_lock = False
    saturation = 0
    contrast = 0
    brightness = 0
    sharpness = 0
    luma_denoise = 0
    chroma_denoise = 0
    control = 'none'

    while True:
        previewFrames = previewQueue.tryGetAll()
        for previewFrame in previewFrames:
            cv2.imshow('preview', previewFrame.getData().reshape(previewFrame.getHeight(), previewFrame.getWidth(), 3))
            #print(previewFrame.getLensPosition(), previewFrame.getExposureTime())

        videoFrames = videoQueue.tryGetAll()
        for videoFrame in videoFrames:
            # Decode JPEG
            if videoMjpeg:
                frame = cv2.imdecode(videoFrame.getData(), cv2.IMREAD_UNCHANGED)
            else:
                frame = videoFrame.getCvFrame()
            # Display
            cv2.imshow('video', frame)

            # Send new cfg to camera
            if sendCamConfig:
                cfg = dai.ImageManipConfig()
                cfg.setCropRect(cropX, cropY, 0, 0)
                configQueue.send(cfg)
                print('Sending new crop - x: ', cropX, ' y: ', cropY)
                sendCamConfig = False

        stillFrames = stillQueue.tryGetAll()
        for stillFrame in stillFrames:
            # Decode JPEG
            frame = cv2.imdecode(stillFrame.getData(), cv2.IMREAD_UNCHANGED)
            # Display
            cv2.imshow('still', frame)

        # Update screen (1ms pooling rate)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            ctrl = dai.CameraControl()
            ctrl.setCaptureStill(True)
            controlQueue.send(ctrl)
        elif key == ord('t'):
            print("Autofocus trigger (and disable continuous)")
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
            ctrl.setAutoFocusTrigger()
            controlQueue.send(ctrl)
        elif key == ord('f'):
            print("Autofocus enable, continuous")
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
            controlQueue.send(ctrl)
        elif key == ord('e'):
            print("Autoexposure enable")
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureEnable()
            controlQueue.send(ctrl)
        elif key == ord('b'):
            print("Auto white-balance enable")
            ctrl = dai.CameraControl()
            ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
            controlQueue.send(ctrl)
        elif key in [ord(','), ord('.')]:
            if key == ord(','): lensPos -= LENS_STEP
            if key == ord('.'): lensPos += LENS_STEP
            lensPos = clamp(lensPos, lensMin, lensMax)
            print("Setting manual focus, lens position: ", lensPos)
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(lensPos)
            controlQueue.send(ctrl)
        elif key in [ord('i'), ord('o'), ord('k'), ord('l')]:
            if key == ord('i'): expTime -= EXP_STEP
            if key == ord('o'): expTime += EXP_STEP
            if key == ord('k'): sensIso -= ISO_STEP
            if key == ord('l'): sensIso += ISO_STEP
            expTime = clamp(expTime, expMin, expMax)
            sensIso = clamp(sensIso, sensMin, sensMax)
            print("Setting manual exposure, time: ", expTime, "iso: ", sensIso)
            ctrl = dai.CameraControl()
            ctrl.setManualExposure(expTime, sensIso)
            controlQueue.send(ctrl)
        elif key in [ord('['), ord(']')]:
            if key == ord('['): wbManual -= WB_STEP
            if key == ord(']'): wbManual += WB_STEP
            wbManual = clamp(wbManual, wbMin, wbMax)
            print("Setting manual white balance, temperature: ", wbManual, "K")
            ctrl = dai.CameraControl()
            ctrl.setManualWhiteBalance(wbManual)
            controlQueue.send(ctrl)
        elif key in [ord('w'), ord('a'), ord('s'), ord('d')]:
            if key == ord('a'):
                cropX = cropX - (maxCropX / camRgb.getResolutionWidth()) * STEP_SIZE
                if cropX < 0: cropX = maxCropX
            elif key == ord('d'):
                cropX = cropX + (maxCropX / camRgb.getResolutionWidth()) * STEP_SIZE
                if cropX > maxCropX: cropX = 0
            elif key == ord('w'):
                cropY = cropY - (maxCropY / camRgb.getResolutionHeight()) * STEP_SIZE
                if cropY < 0: cropY = maxCropY
            elif key == ord('s'):
                cropY = cropY + (maxCropY / camRgb.getResolutionHeight()) * STEP_SIZE
                if cropY > maxCropY: cropY = 0
            sendCamConfig = True
        elif key == ord('1'):
            awb_lock = not awb_lock
            print("Auto white balance lock:", awb_lock)
            ctrl = dai.CameraControl()
            ctrl.setAutoWhiteBalanceLock(awb_lock)
            controlQueue.send(ctrl)
        elif key == ord('2'):
            ae_lock = not ae_lock
            print("Auto exposure lock:", ae_lock)
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureLock(ae_lock)
            controlQueue.send(ctrl)
        elif key >= 0 and chr(key) in '34567890[]':
            if   key == ord('3'): control = 'awb_mode'
            elif key == ord('4'): control = 'ae_comp'
            elif key == ord('5'): control = 'anti_banding_mode'
            elif key == ord('6'): control = 'effect_mode'
            elif key == ord('7'): control = 'brightness'
            elif key == ord('8'): control = 'contrast'
            elif key == ord('9'): control = 'saturation'
            elif key == ord('0'): control = 'sharpness'
            elif key == ord('['): control = 'luma_denoise'
            elif key == ord(']'): control = 'chroma_denoise'
            print("Selected control:", control)
        elif key in [ord('-'), ord('_'), ord('+'), ord('=')]:
            change = 0
            if key in [ord('-'), ord('_')]: change = -1
            if key in [ord('+'), ord('=')]: change = 1
            ctrl = dai.CameraControl()
            if control == 'none':
                print("Please select a control first using keys 3..9 0 [ ]")
            elif control == 'ae_comp':
                ae_comp = clamp(ae_comp + change, -9, 9)
                print("Auto exposure compensation:", ae_comp)
                ctrl.setAutoExposureCompensation(ae_comp)
            elif control == 'anti_banding_mode':
                cnt = len(anti_banding_mode_list)
                anti_banding_mode_idx = (anti_banding_mode_idx + cnt + change) % cnt
                anti_banding_mode = anti_banding_mode_list[anti_banding_mode_idx]
                print("Anti-banding mode:", anti_banding_mode)
                ctrl.setAntiBandingMode(anti_banding_mode)
            elif control == 'awb_mode':
                cnt = len(awb_mode_list)
                awb_mode_idx = (awb_mode_idx + cnt + change) % cnt
                awb_mode = awb_mode_list[awb_mode_idx]
                print("Auto white balance mode:", awb_mode)
                ctrl.setAutoWhiteBalanceMode(awb_mode)
            elif control == 'effect_mode':
                cnt = len(effect_mode_list)
                effect_mode_idx = (effect_mode_idx + cnt + change) % cnt
                effect_mode = effect_mode_list[effect_mode_idx]
                print("Effect mode:", effect_mode)
                ctrl.setEffectMode(effect_mode)
            elif control == 'brightness':
                brightness = clamp(brightness + change, -10, 10)
                print("Brightness:", brightness)
                ctrl.setBrightness(brightness)
            elif control == 'contrast':
                contrast = clamp(contrast + change, -10, 10)
                print("Contrast:", contrast)
                ctrl.setContrast(contrast)
            elif control == 'saturation':
                saturation = clamp(saturation + change, -10, 10)
                print("Saturation:", saturation)
                ctrl.setSaturation(saturation)
            elif control == 'sharpness':
                sharpness = clamp(sharpness + change, 0, 4)
                print("Sharpness:", sharpness)
                ctrl.setSharpness(sharpness)
            elif control == 'luma_denoise':
                luma_denoise = clamp(luma_denoise + change, 0, 4)
                print("Luma denoise:", luma_denoise)
                ctrl.setLumaDenoise(luma_denoise)
            elif control == 'chroma_denoise':
                chroma_denoise = clamp(chroma_denoise + change, 0, 4)
                print("Chroma denoise:", chroma_denoise)
                ctrl.setChromaDenoise(chroma_denoise)
            controlQueue.send(ctrl)
