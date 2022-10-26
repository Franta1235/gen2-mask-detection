from contextlib import ExitStack
from datetime import timedelta
import depthai as dai
import numpy as np
import blobconverter
from robothub_sdk import App, IS_INTERACTIVE, Config
from typing import List, Generator
from robothub_sdk.device import Device
from MultiMsgSync import TwoStageHostSeqSync

# IS_INTERACTIVE = False
if IS_INTERACTIVE:
    import cv2


def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())


class MaskDetection(App):
    def on_initialize(self, devices: List[dai.DeviceInfo]):
        self.msgs = {}
        self.detect_threshold = 0.5

    def on_configuration(self, old_configuration: Config):
        print("Configuration update", self.config.values())
        self.detect_threshold = self.config.detect_threshold

    def on_setup(self, device: Device):
        self.stereo = 1 < len(device.cameras)
        device.pipeline = self.create_pipeline(self.stereo)
        self.sync = TwoStageHostSeqSync()
        self.queues = {}

    def on_update(self):
        if IS_INTERACTIVE:
            for device in self.devices:
                for name, q in self.queues.items():
                    # Add all msgs (color frames, object detections and recognitions) to the Sync class.
                    if q.has():
                        self.sync.add_msg(q.get(), name)

                msgs = self.sync.get_msgs()
                if msgs is not None:
                    frame = msgs["color"].getCvFrame()
                    detections = msgs["detection"].detections
                    recognitions = msgs["recognition"]

                    for i, detection in enumerate(detections):
                        bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

                        # Decoding of recognition results
                        rec = recognitions[i].getFirstLayerFp16()
                        index = np.argmax(log_softmax(rec))
                        text = "No Mask"
                        color = (0, 0, 255)  # Red
                        if index == 1:
                            text = "Mask"
                            color = (0, 255, 0)

                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
                        y = (bbox[1] + bbox[3]) // 2
                        cv2.putText(frame, text, (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0), 8)
                        cv2.putText(frame, text, (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 2)
                        if self.stereo:
                            # You could also get detection.spatialCoordinates.x and detection.spatialCoordinates.y coordinates
                            coords = "Z: {:.2f} m".format(detection.spatialCoordinates.z / 1000)
                            cv2.putText(frame, coords, (bbox[0], y + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 8)
                            cv2.putText(frame, coords, (bbox[0], y + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)

                    cv2.imshow("Camera", frame)
                if cv2.waitKey(1) == ord('q'):
                    self.stop()
        else:
            # Report in Robothub console
            for device in self.devices:
                for name, q in self.queues.items():
                    # Add all msgs (color frames, object detections and recognitions) to the Sync class.
                    if q.has():
                        self.sync.add_msg(q.get(), name)

                msgs = self.sync.get_msgs()
                if msgs is not None:
                    detections = msgs["detection"].detections
                    recognitions = msgs["recognition"]

                    for i, detection in enumerate(detections):
                        # Decoding of recognition results
                        rec = recognitions[i].getFirstLayerFp16()
                        index = np.argmax(log_softmax(rec))
                        text = "No Mask"
                        if index == 1:
                            text = "Mask"
                        print(f"{name}-{i} {text}")

    def create_pipeline(self, stereo):
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

        # ImageManip will resize the frame before sending it to the Face detection NN node
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
            print("OAK-D detected, app will display spatial coordinates")
            face_det_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
            face_det_nn.setBoundingBoxScaleFactor(0.8)
            face_det_nn.setDepthLowerThreshold(100)
            face_det_nn.setDepthUpperThreshold(5000)
            stereo.depth.link(face_det_nn.inputDepth)
        else:  # Detection network if OAK-1
            print("OAK-1 detected, app won't display spatial coordinates")
            face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)

        face_det_nn.setConfidenceThreshold(self.detect_threshold)
        face_det_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
        face_det_manip.out.link(face_det_nn.input)

        # Send face detections to the host (for bounding boxes)
        face_det_xout = pipeline.create(dai.node.XLinkOut)
        face_det_xout.setStreamName("detection")
        face_det_nn.out.link(face_det_xout.input)

        # Script node will take the output from the face detection NN as an input and set ImageManipConfig
        # to the 'recognition_manip' to crop the initial frame
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
                    # node.warn(f"Sending {i + 1}. det. Seq {seq}. Det {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
                    cfg.setResize(224, 224)
                    cfg.setKeepAspectRatio(False)
                    node.io['manip_cfg'].send(cfg)
                    node.io['manip_img'].send(img)
        """)

        recognition_manip = pipeline.create(dai.node.ImageManip)
        recognition_manip.initialConfig.setResize(224, 224)
        recognition_manip.setWaitForConfigInput(True)
        image_manip_script.outputs['manip_cfg'].link(recognition_manip.inputConfig)
        image_manip_script.outputs['manip_img'].link(recognition_manip.inputImage)

        # Second stange recognition NN
        print("Creating recognition Neural Network...")
        recognition_nn = pipeline.create(dai.node.NeuralNetwork)
        recognition_nn.setBlobPath(blobconverter.from_zoo(name="sbd_mask_classification_224x224", zoo_type="depthai", shaves=6))
        recognition_manip.out.link(recognition_nn.input)

        recognition_xout = pipeline.create(dai.node.XLinkOut)
        recognition_xout.setStreamName("recognition")
        recognition_nn.out.link(recognition_xout.input)

        return pipeline

    def _run_pipelines(self, devices: List[dai.DeviceInfo]) -> Generator[bool, None, None]:
        for device in self.devices:
            device.internal.close()
        self.devices.clear()

        openvino_version = dai.OpenVINO.Version.VERSION_2021_4
        usb2_mode = False
        self.on_initialize(devices)

        with ExitStack() as stack:
            for device_info in devices:
                dai_device: dai.Device = stack.enter_context(dai.Device(openvino_version, device_info, usb2_mode))
                device_id = dai_device.getMxId()
                configuration = self._device_configuration.get(device_id, None)
                device = Device(device_id, device_info, dai_device, configuration)

                print(f"=== Connected to: {device_id}")
                print(str(device))
                self.on_setup(device)

                dai_device.startPipeline(device.pipeline)

                dai_device.setLogLevel(dai.LogLevel.WARN)
                dai_device.setLogOutputLevel(dai.LogLevel.WARN)

                """
                self._comm.report_device(device)
                for stream in device.streams.inputs():
                    queue = dai_device.getInputQueue(stream.input_queue_name, maxSize=1, blocking=False)
                    stream.register_queue(queue)
                """

                """
                for stream in device.streams.outputs():
                    print(stream.published.output_queue_name)
                    consumed_queue = dai_device.getOutputQueue(stream.output_queue_name, maxSize=2, blocking=False) if stream.is_consumed else None
                    published_queue = None
                    if stream.is_published:
                        published_queue = consumed_queue
                        if stream.published.output_queue_name != stream.output_queue_name:
                            published_queue = dai_device.getOutputQueue(stream.published.output_queue_name, maxSize=2, blocking=False)

                    if published_queue:
                        if stream.published.type == StreamType.ENCODED:
                            self._comm.add_video_stream(stream.published)
                            published_queue.addCallback(partial(self._comm.send_stream, stream.published))
                        elif stream.published.type == StreamType.STATISTICS:
                            published_queue.addCallback(partial(self._comm.send_statistics, device_id))
                        else:
                            raise RobotHubFatalException("Published stream is not supported")

                    if consumed_queue:
                        if stream.rate > 0:
                            self._min_update_rate = min(self._min_update_rate, 1 / stream.rate)
                        consumed_queue.addCallback(stream.queue_callback)
                """

                for name in ["color", "detection", "recognition"]:
                    self.queues[name] = dai_device.getOutputQueue(name)

                self.devices.append(device)

            while self.running:
                self.on_update()

            """
            self._comm.report_online()
            last_run = 0
            while self.running:
                had_items = False
                now = time.monotonic()
                for device in self.devices:
                    # NOTE: needs depthai>=2.17.4
                    if device.internal.isClosed():
                        raise RuntimeError(f"Device {device.id} / {device.name} disconnected.")
                    last_item = 0
                    for stream in device.streams.outputs():
                        last_item = max(last_item, stream.last_timestamp)
                        if stream.last_timestamp > last_run:
                            had_items = True
                    if last_item > 0 and now - last_item > STREAM_STUCK_TIMEOUT:
                        # NOTE(michal) temporary measure before we figure out how to fix this
                        raise RuntimeError(f"Device {device.id} / {device.name} stuck. Last message received {now - last_item}s ago.")

                if had_items:
                    self.on_update()
                last_run = now
                yield had_items
            """


app = MaskDetection()
app.run()
