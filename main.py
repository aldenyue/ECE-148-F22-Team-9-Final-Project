from MultiMsgSync import TwoStageHostSeqSync
import blobconverter
import cv2
import depthai as dai
import numpy as np
import time
from playsound import playsound
import pyvesc
import threading

# make sure you install all of the libraries after installing the requirements

def play_music(music):
    threading.Thread(target=playsound, args=(music,), daemon=True).start()

class VESC:
    ''' 
    VESC Motor controler using pyvesc
    This is used for most electric scateboards.
    
    inputs: serial_port---- port used communicate with vesc. for linux should be something like /dev/ttyACM1
    has_sensor=False------- default value from pyvesc
    start_heartbeat=True----default value from pyvesc (I believe this sets up a heartbeat and kills speed if lost)
    baudrate=115200--------- baudrate used for communication with VESC
    timeout=0.05-------------time it will try before giving up on establishing connection
    
    percent=.2--------------max percentage of the dutycycle that the motor will be set to
    outputs: none
    
    uses the pyvesc library to open communication with the VESC and sets the servo to the angle (0-1) and the duty_cycle(speed of the car) to the throttle (mapped so that percentage will be max/min speed)
    
    Note that this depends on pyvesc, but using pip install pyvesc will create a pyvesc file that
    can only set the speed, but not set the servo angle. 
    
    Instead please use:
    pip install git+https://github.com/LiamBindle/PyVESC.git@master
    to install the pyvesc library
    '''
    def __init__(self, serial_port, percent=.2, has_sensor=False, start_heartbeat=True, baudrate=115200, timeout=0.05, steering_scale = 1.0, steering_offset = 0.0 ):
        
        try:
            import pyvesc
        except Exception as err:
            print("\n\n\n\n", err, "\n")
            print("please use the following command to import pyvesc so that you can also set")
            print("the servo position:")
            print("pip install git+https://github.com/LiamBindle/PyVESC.git@master")
            print("\n\n\n")
            time.sleep(1)
            raise
        
        assert percent <= 1 and percent >= -1,'\n\nOnly percentages are allowed for MAX_VESC_SPEED (we recommend a value of about .2) (negative values flip direction of motor)'
        self.steering_scale = steering_scale
        self.steering_offset = steering_offset
        self.percent = percent
        
        try:
            self.v = pyvesc.VESC(serial_port, has_sensor, start_heartbeat, baudrate, timeout)
        except Exception as err:
            print("\n\n\n\n", err)
            print("\n\nto fix permission denied errors, try running the following command:")
            print("sudo chmod a+rw {}".format(serial_port), "\n\n\n\n")
            time.sleep(1)
            raise
        
    def run(self, angle, throttle):
        self.v.set_servo((angle * self.steering_scale) + self.steering_offset)
        self.v.set_duty_cycle(throttle*self.percent)


Vesc_object = VESC('/dev/ttyACM0')
# Straighten Car out
Vesc_object.run(0.5, 0)
def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']

def create_pipeline(stereo):
    pipeline = dai.Pipeline()

    print("Creating Color Camera...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(1080, 1080)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
 

    cam_xout = pipeline.create(dai.node.XLinkOut)
    cam_xout.setStreamName("color")
    cam.preview.link(cam_xout.input)

    # Workaround: remove in 2.18, use `cam.setPreviewNumFramesPool(10)`
    # This manip uses 15*3.5 MB => 52 MB of RAM.
    copy_manip = pipeline.create(dai.node.ImageManip)
    copy_manip.setNumFramesPool(15)
    copy_manip.setMaxOutputFrameSize(3499200)
    cam.preview.link(copy_manip.inputImage)

    # ImageManip that will crop the frame before sending it to the Face detection NN node
    face_det_manip = pipeline.create(dai.node.ImageManip)
    face_det_manip.initialConfig.setResize(300, 300)
    face_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
    copy_manip.out.link(face_det_manip.inputImage)

    if stereo:
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

        monoRight = pipeline.create(dai.node.MonoCamera)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        # Spatial Detection network if OAK-D
        print("OAK-D detected, app will display spatial coordiantes")
        face_det_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        face_det_nn.setBoundingBoxScaleFactor(0.8)
        face_det_nn.setDepthLowerThreshold(100)
        face_det_nn.setDepthUpperThreshold(5000)
        stereo.depth.link(face_det_nn.inputDepth)
    else: # Detection network if OAK-1
        print("OAK-1 detected, app won't display spatial coordiantes")
        face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)

    face_det_nn.setConfidenceThreshold(0.5)
    face_det_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
    face_det_manip.out.link(face_det_nn.input)

    # Send face detections to the host (for bounding boxes)
    face_det_xout = pipeline.create(dai.node.XLinkOut)
    face_det_xout.setStreamName("detection")
    face_det_nn.out.link(face_det_xout.input)

    # Script node will take the output from the face detection NN as an input and set ImageManipConfig
    # to the 'age_gender_manip' to crop the initial frame
    image_manip_script = pipeline.create(dai.node.Script)
    face_det_nn.out.link(image_manip_script.inputs['face_det_in'])

    # Only send metadata, we are only interested in timestamp, so we can sync
    # depth frames with NN output
    face_det_nn.passthrough.link(image_manip_script.inputs['passthrough'])
    copy_manip.out.link(image_manip_script.inputs['preview'])

    image_manip_script.setScript("""
    import time
    msgs = dict()

    def add_msg(msg, name, seq = None):
        global msgs
        if seq is None:
            seq = msg.getSequenceNum()
        seq = str(seq)
        # node.warn(f"New msg {name}, seq {seq}")

        # Each seq number has it's own dict of msgs
        if seq not in msgs:
            msgs[seq] = dict()
        msgs[seq][name] = msg

        # To avoid freezing (not necessary for this ObjDet model)
        if 15 < len(msgs):
            node.warn(f"Removing first element! len {len(msgs)}")
            msgs.popitem() # Remove first element

    def get_msgs():
        global msgs
        seq_remove = [] # Arr of sequence numbers to get deleted
        for seq, syncMsgs in msgs.items():
            seq_remove.append(seq) # Will get removed from dict if we find synced msgs pair
            # node.warn(f"Checking sync {seq}")

            # Check if we have both detections and color frame with this sequence number
            if len(syncMsgs) == 2: # 1 frame, 1 detection
                for rm in seq_remove:
                    del msgs[rm]
                # node.warn(f"synced {seq}. Removed older sync values. len {len(msgs)}")
                return syncMsgs # Returned synced msgs
        return None

    def correct_bb(bb):
        if bb.xmin < 0: bb.xmin = 0.001
        if bb.ymin < 0: bb.ymin = 0.001
        if bb.xmax > 1: bb.xmax = 0.999
        if bb.ymax > 1: bb.ymax = 0.999
        return bb

    while True:
        time.sleep(0.001) # Avoid lazy looping

        preview = node.io['preview'].tryGet()
        if preview is not None:
            add_msg(preview, 'preview')

        face_dets = node.io['face_det_in'].tryGet()
        if face_dets is not None:
            # TODO: in 2.18.0.0 use face_dets.getSequenceNum()
            passthrough = node.io['passthrough'].get()
            seq = passthrough.getSequenceNum()
            add_msg(face_dets, 'dets', seq)

        sync_msgs = get_msgs()
        if sync_msgs is not None:
            img = sync_msgs['preview']
            dets = sync_msgs['dets']
            for i, det in enumerate(dets.detections):
                cfg = ImageManipConfig()
                correct_bb(det)
                cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
                # node.warn(f"Sending {i + 1}. age/gender det. Seq {seq}. Det {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
                cfg.setResize(64, 64)
                cfg.setKeepAspectRatio(False)
                node.io['manip_cfg'].send(cfg)
                node.io['manip_img'].send(img)
    """)

    manip_manip = pipeline.create(dai.node.ImageManip)
    manip_manip.initialConfig.setResize(64, 64)
    manip_manip.setWaitForConfigInput(True)
    image_manip_script.outputs['manip_cfg'].link(manip_manip.inputConfig)
    image_manip_script.outputs['manip_img'].link(manip_manip.inputImage)

    # This ImageManip will crop the mono frame based on the NN detections. Resulting image will be the cropped
    # face that was detected by the face-detection NN.
    emotions_nn = pipeline.create(dai.node.NeuralNetwork)
    emotions_nn.setBlobPath(blobconverter.from_zoo(name="emotions-recognition-retail-0003", shaves=6))
    manip_manip.out.link(emotions_nn.input)

    recognition_xout = pipeline.create(dai.node.XLinkOut)
    recognition_xout.setStreamName("recognition")
    emotions_nn.out.link(recognition_xout.input)

    return pipeline

input('Press Enter to Begin Therapy Mode...')

print('Searching for Depression')

with dai.Device() as device:
    stereo = 1 < len(device.getConnectedCameras())
    device.startPipeline(create_pipeline(stereo))

    sync = TwoStageHostSeqSync()
    queues = {}
    # Create output queues
    for name in ["color", "detection", "recognition"]:
        queues[name] = device.getOutputQueue(name)

    sad_count = 0
    x = 0
    y = 0
    z = 0
    while True:
        for name, q in queues.items():
            # Add all msgs (color frames, object detections and age/gender recognitions) to the Sync class.
            if q.has():
                sync.add_msg(q.get(), name)
        
        msgs = sync.get_msgs()
        emotion_name = 'neutral'
        

        if msgs is not None:
            frame = msgs["color"].getCvFrame()
            detections = msgs["detection"].detections
            recognitions = msgs["recognition"]

            for i, detection in enumerate(detections):
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

                rec = recognitions[i]

                emotion_results = np.array(rec.getFirstLayerFp16())
                emotion_name = emotions[np.argmax(emotion_results)]
                if emotion_name == 'sad':
                    print('Possible Sad boi detected')
                    sad_count = sad_count + 1
                    print(sad_count)
                
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                y = (bbox[1] + bbox[3]) // 2
                x = (bbox[0] + bbox[2]) // 2
                cv2.circle(frame, (x, y), 7, (255, 255, 255), -1)
                cv2.line(frame,(x, y),(540, 1080),(0,0,255),5)
                cv2.putText(frame, emotion_name, (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0), 8)
                cv2.putText(frame, emotion_name, (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 2)
                if stereo:
                    # You could also get detection.spatialCoordinates.x and detection.spatialCoordinates.y coordinates
                    z= detection.spatialCoordinates.z/1000
                    coords = "Z: {:.2f} m".format(detection.spatialCoordinates.z/1000)
                    cv2.putText(frame, coords, (bbox[0], y + 35), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 8)
                    cv2.putText(frame, coords, (bbox[0], y + 35), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
            frame = cv2.resize(frame,(650,500))
            cv2.imshow("Camera", frame)
        if  sad_count > 20 or cv2.waitKey(1) == ord('q'):
            print('Sadness Threshold Reached...')
            print('Beginning Therapeutic Treatment')
            print('Sadness Detected at:' + str(x) + ', ' + str(y))

            # proportion to convert frame difference to steering input
            # Specifications: range ~ 2.5-5 meters,  FOV: 40 degrees

            # FOV angle is from 70 - 110 degrees
            # (y - 540)/5 

            # Need to tune steering offset and proportional constant (also consider a seperate constant for left and right turns) varies from car to car
            offset = -0.02
            p = 0.6

            # turn input based on distance from the center of the image
            turn_input = 0.5 + p*(x-540)/(1080*5) - offset

            print('turn input: ' + str(turn_input))
            drive_time = z*5/3

            Vesc_object.run(turn_input, 0.1)

            time.sleep(drive_time)
            
            Vesc_object.run(0.5, 0)

            play_music('brightside.mp3')

            time.sleep(3)

            # Dance Time :)
            for i in range(300):
                Vesc_object.run(0.25, 0)
                time.sleep(1)
                Vesc_object.run(0.75, 0)
                time.sleep(1)

            break
