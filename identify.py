import sensor, image, time, os, tf, uos, gc, math, lcd
from pyb import UART

# Reset sensor
sensor.reset()
sensor.set_vflip(1)
sensor.set_hmirror(1)

# Sensor settings
sensor.set_contrast(3)
sensor.set_gainceiling(16)
# QQVGA and GRAYSCALE are the best for face tracking.
sensor.set_framesize(sensor.QQVGA)  # Reduce frame size to QQVGA (160x120)
sensor.set_pixformat(sensor.RGB565)
sensor.set_windowing((160, 120))     # Set 160x120 window.
sensor.skip_frames(time=2000)        # Let the camera adjust.

lcd.init()

net_face_identify = None
net_person_posture = None
face_identify = None
person_posture = None
min_confidence = 0.5

# Load Haar Cascade
face_cascade = image.HaarCascade("frontalface", stages=25)
print(face_cascade)

try:
    net_face_identify = tf.load("face_identify.tflite", load_to_fb=uos.stat('face_identify.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    print(e)
    raise Exception('Failed to load "face_identify.tflite", did you copy the .tflite and face_identify.txt file onto the mass-storage device? (' + str(e) + ')')

try:
    face_identify = [line.rstrip('\n') for line in open("face_identify.txt")]
except Exception as e:
    raise Exception('Failed to load "face_identify.txt", did you copy the .tflite and face_identify.txt file onto the mass-storage device? (' + str(e) + ')')

try:
    net_person_posture = tf.load("person_posture.tflite", load_to_fb=uos.stat('person_posture.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    raise Exception('Failed to load "person_posture.tflite", did you copy the .tflite and person_posture.txt file onto the mass-storage device? (' + str(e) + ')')

try:
    person_posture = [line.rstrip('\n') for line in open("person_posture.txt")]
except Exception as e:
    raise Exception('Failed to load "person_posture.txt", did you copy the .tflite and person_posture.txt file onto the mass-storage device? (' + str(e) + ')')

colors = [
    (255,   0,   0),
    (  0, 255,   0),
    (255, 255,   0),
    (  0,   0, 255),
    (255,   0, 255),
    (  0, 255, 255),
    (255, 255, 255),
]
emotion = ["happy","sad"]
a = " "
# FPS clock
clock = time.clock()

def compare_floats(arr1, arr2, tolerance=1e-9):
    if abs(arr1 - arr2) < tolerance:
        return 0  # 两个数在允许的误差范围内相等
    elif arr1 < arr2:
        return -1
    else:
        return 1

uart = UART(2, 0, timeout_char=1000)

while True:
    clock.tick()

    # Capture snapshot
    img = sensor.snapshot()

    # Find objects.
    objects = img.find_features(face_cascade, threshold=0.75, scale_factor=1.25)

    # Draw objects
    for r in objects:
        uart.write('p')
        print("*******person*********")
        img.draw_rectangle(r)

        # Crop the face region for emotion detection
        #face_img = img.to_grayscale()
        face_img = img.copy(roi=r)

        for obj in net_face_identify.classify(face_img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
            print("**********\nPredictions at [x=%d,y=%d,w=%d,h=%d]" % obj.rect())
            img.draw_rectangle(obj.rect())
            predictions_list = list(zip(face_identify, obj.output()))
            for j in range(len(predictions_list)):
                print("%s = %f" % (predictions_list[j][0], predictions_list[j][1]))

            for i in range(1, len(predictions_list)):
                array1 = predictions_list[i-1][1]
                array2 = predictions_list[i][1]
                result = compare_floats(array1, array2)
                if result == 0:
                    print(f"{array1} 等于 {array2}")
                elif result < 0:
                    a = "sad"
                    uart.write('s')
                    print("********** %s **********" % emotion[1])
                else:
                    a = "happy"
                    uart.write('h')
                    print("********** %s **********" % emotion[0])

    # Detect posture
    for i, detection_list in enumerate(net_person_posture.detect(img, thresholds=[(math.ceil(min_confidence * 255), 255)])):
        if (i == 0): continue # background class
        if (len(detection_list) == 0): continue # no detections for this class?

        if (i == 0):
            uart.write('t')
        else:
            uart.write('l')
        print("********** %s **********" % person_posture[i])

        for d in detection_list:
            [x, y, w, h] = d.rect()
            center_x = math.floor(x + (w / 2))
            center_y = math.floor(y + (h / 2))
            print('x %d\ty %d' % (center_x, center_y))
            img.draw_circle((center_x, center_y, 12), color=colors[i], thickness=2)
    # Print FPS.
    print(clock.fps())
    img.draw_strings(0, 0, "fps:%2.2f" % clock.fps(), color=(255,0,0), scale=1)
    img.draw_string(7, 7, "state:%s" % person_posture[i-1], color=(0,255,0), scale=1)
    img.draw_string(13, 13,"emotion:%s" % a, color=(0,0,255), scale=1)
    lcd.display(img)  # Take a picture and display the image.
