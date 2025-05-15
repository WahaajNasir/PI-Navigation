import cv2 as cv
import numpy as np

# ─── HSV THRESHOLDS ───────────────────────────────────────────────────────────
SUNNY_LOWER = np.array([16.73, 30.97, 182.68])
SUNNY_UPPER = np.array([19.89, 32.53, 201.86])
CLOUDY_LOWER = np.array([49.84,  7.70, 189.92])
CLOUDY_UPPER = np.array([157.30,16.50, 204.29])

# Precomputed means for HSV segmentation
SUNNY_MEAN  = np.array([18.31, 31.75, 192.27])
SHADOW_MEAN = np.array([94.00, 29.20, 106.75])

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def contrast_stretching(img):
    mi, ma = img.min(), img.max()
    return ((img - mi) * (255.0/(ma - mi))).astype(np.uint8)

def kmeans_segmentation_fast(gray, k=3):
    Z = gray.reshape(-1,1).astype(np.float32)
    crit = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER,10,1.0)
    _,lbl,_ = cv.kmeans(Z, k, None, crit, 5, cv.KMEANS_RANDOM_CENTERS)
    lbl = lbl.reshape(gray.shape)
    h,w = lbl.shape
    guide = lbl[int(0.85*h):, int(0.4*w):int(0.6*w)]
    counts = [np.sum(guide==i) for i in range(k)]
    road_lbl = np.argmax(counts)
    return np.where(lbl==road_lbl,255,127).astype(np.uint8)

def get_adaptive_range(center, margin):
    lo = max(0, center*(1-margin)); hi = min(255, center*(1+margin))
    return int(lo), int(hi)

def hsv_segmentation(frame_hsv):
    small = cv.resize(frame_hsv, (0,0), fx=0.25, fy=0.25, interpolation=cv.INTER_AREA)
    hlo,hhi = get_adaptive_range(SUNNY_MEAN[0],0.25)
    slo,shi = get_adaptive_range(SUNNY_MEAN[1],0.25)
    vlo,vhi = get_adaptive_range(SUNNY_MEAN[2],0.25)
    sunny = cv.inRange(small, (hlo,slo,vlo), (hhi,shi,vhi))
    hlo,hhi = get_adaptive_range(SHADOW_MEAN[0],0.30)
    slo,shi = get_adaptive_range(SHADOW_MEAN[1],0.30)
    vlo,vhi = get_adaptive_range(SHADOW_MEAN[2],0.30)
    shadow = cv.inRange(small, (hlo,slo,vlo), (hhi,shi,vhi))
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    sunny_cl = cv.morphologyEx(sunny, cv.MORPH_CLOSE, k)
    dist = cv.distanceTransform(sunny_cl, cv.DIST_L2,5)
    shadow_cl = cv.bitwise_and(shadow, (dist<50).astype(np.uint8)*255)
    comb = cv.bitwise_or(sunny_cl, shadow_cl)
    return cv.resize(comb, (frame_hsv.shape[1], frame_hsv.shape[0]),
                     interpolation=cv.INTER_NEAREST)

def post_processing(m):
    k = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    m = cv.morphologyEx(m, cv.MORPH_OPEN, k)
    return cv.morphologyEx(m, cv.MORPH_CLOSE, k)

def remove_small_objects(m, min_size):
    n,lbl,stats,_ = cv.connectedComponentsWithStats(m,8,cv.CV_32S)
    out = np.zeros_like(m)
    for i in range(1,n):
        if stats[i,cv.CC_STAT_AREA]>=min_size:
            out[lbl==i]=255
    return out

def remove_small_zero_objects(m, min_size):
    inv = cv.bitwise_not(m)
    n,lbl,stats,_ = cv.connectedComponentsWithStats(inv,8,cv.CV_32S)
    out = np.ones_like(m)*255
    for i in range(1,n):
        if stats[i,cv.CC_STAT_AREA]>=min_size:
            out[lbl==i]=0
    return out

def has_clear_gap(band, min_width=200):
    for row in band:
        cnt=0
        for v in (row==255):
            cnt = cnt+1 if v else 0
            if cnt>=min_width:
                return True
    return False

def classify_frame(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    avg = np.mean(hsv.reshape(-1,3), axis=0)
    if ((SUNNY_LOWER <= avg) & (avg <= SUNNY_UPPER)).all(): return "Sunny"
    if ((CLOUDY_LOWER <= avg) & (avg <= CLOUDY_UPPER)).all(): return "Cloudy"
    return "Unknown"

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
cap = cv.VideoCapture(r"D:\Uni\Semester 6\DIP\Self\Project\Tesla_Model_Sangi\Dataset\Cloudy\PXL_20250325_043922504.TS.mp4")
detect_obj_count = 0
road_classification_history = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv.resize(frame, (1280,720), interpolation=cv.INTER_AREA)
    cond  = classify_frame(frame)

    # ── SEGMENTATION ──────────────────────────────────────────────────────────
    if cond == "Sunny":
        seg = hsv_segmentation(cv.cvtColor(frame, cv.COLOR_BGR2HSV))
    else:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cont = contrast_stretching(gray)
        mb   = cv.medianBlur(cont,5)
        ed   = cv.Canny(mb,50,150)
        edc  = cv.morphologyEx(ed,cv.MORPH_CLOSE,
                cv.getStructuringElement(cv.MORPH_RECT,(5,5)))
        tmp, cnts = mb.copy(), cv.findContours(edc,cv.RETR_EXTERNAL,
                             cv.CHAIN_APPROX_SIMPLE)[0]
        cv.drawContours(tmp,cnts,-1,0,cv.FILLED)
        down = cv.resize(tmp,(0,0),fx=0.25,fy=0.25,interpolation=cv.INTER_AREA)
        seg  = cv.resize(kmeans_segmentation_fast(down),
                        gray.shape[::-1], interpolation=cv.INTER_NEAREST)

    # ── POST-PROCESS & CLEANUP ─────────────────────────────────────────────────
    seg = post_processing(seg)
    seg = remove_small_objects(seg,25000)
    seg = remove_small_zero_objects(seg,5000)

    # ── UNKNOWN‐FALLBACK ────────────────────────────────────────────────────────
    cov = np.sum(seg==255)/seg.size
    if cond=="Unknown" and cov>0.8:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cont = contrast_stretching(gray)
        mb   = cv.medianBlur(cont,5)
        ed   = cv.Canny(mb,50,150)
        edc  = cv.morphologyEx(ed,cv.MORPH_CLOSE,
                cv.getStructuringElement(cv.MORPH_RECT,(5,5)))
        tmp, cnts = mb.copy(), cv.findContours(edc,cv.RETR_EXTERNAL,
                             cv.CHAIN_APPROX_SIMPLE)[0]
        cv.drawContours(tmp,cnts,-1,0,cv.FILLED)
        down = cv.resize(tmp,(0,0),fx=0.25,fy=0.25,interpolation=cv.INTER_AREA)
        seg  = cv.resize(kmeans_segmentation_fast(down),
                        gray.shape[::-1], interpolation=cv.INTER_NEAREST)
        seg = post_processing(seg)
        seg = remove_small_objects(seg,25000)
        seg = remove_small_zero_objects(seg,5000)

    road_mask = seg.copy()
    segmented = seg.copy()  # for display

    # ── OVERLAY & DECISION LOGIC ───────────────────────────────────────────────
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    overlay     = cv.bitwise_and(gray, gray, mask=road_mask)
    vis_overlay = cv.cvtColor(overlay.copy(), cv.COLOR_GRAY2BGR)

    h, w = road_mask.shape
    center_x = w//2
    center_y= h//2 + 50
    detection_y = center_y - 50

    top_y    = np.argmax(np.any(road_mask==255,axis=1)) + 50
    bottom_y = h - np.argmax(np.any(road_mask[::-1]==255,axis=1)) - 1

    cv.line(vis_overlay, (0,top_y),    (w,top_y),    (255,0,255),2)
    cv.line(vis_overlay, (0,bottom_y),(w,bottom_y),(0,255,255),2)

    top_px    = np.sum(road_mask[top_y,:]==255)
    bottom_px = np.sum(road_mask[bottom_y,:]==255)

    road = "Straight"
    if bottom_px!=0 and top_px/bottom_px<0.125:
        road_classification_history.append("Sloped")
    else:
        road_classification_history.append("Straight")
        center_y = h//2 + 50

    if len(road_classification_history)>10:
        road_classification_history.pop(0)
    if road_classification_history.count("Sloped")>=7:
        road = "Sloped"
        center_y = bottom_y - 100
        detection_y = center_y - 50

    check_h, check_w = 75, 50
    left_x, right_x = center_x-check_w, center_x+check_w
    vertical_strip  = road_mask[center_y-check_h:center_y, left_x:right_x]
    vertical_clear  = np.all(vertical_strip==255)

    cv.rectangle(vis_overlay,(left_x, center_y-check_h),
                 (right_x, center_y),(0,0,255),2)
    cv.rectangle(vis_overlay,(center_x-200, detection_y-100),
                 (center_x, detection_y+100),(0,255,0),2)
    cv.rectangle(vis_overlay,(center_x, detection_y-100),
                 (center_x+200, detection_y+100),(0,255,0),2)

    middle_line = road_mask[top_y:bottom_y, center_x]
    cv.line(vis_overlay,(center_x,top_y),(center_x,bottom_y),(255,255,0),2)

    middle_roi_w = 100+(detect_obj_count>50)*100
    middle_roi_h = 100
    middle_roi = road_mask[center_y-middle_roi_h:center_y+middle_roi_h,
                           center_x-middle_roi_w//2:center_x+middle_roi_w//2]
    object_detected = np.any(middle_roi==0)
    if object_detected:
        detect_obj_count += 1
    else:
        detect_obj_count = max(detect_obj_count-1,0)

    cv.rectangle(vis_overlay,
        (center_x-middle_roi_w//2, center_y-middle_roi_h),
        (center_x+middle_roi_w//2, center_y+middle_roi_h),
        (255,0,0),2)

    zero_ratio = np.sum(middle_roi==0)/middle_roi.size
    if   detect_obj_count>150: dec="Stopping"
    elif 60<detect_obj_count<=150: dec="Slowing Down"
    else:                          dec="Moving Forward"

    left_band  = road_mask[detection_y-100:detection_y+100, :center_x]
    right_band = road_mask[detection_y-100:detection_y+100, center_x:]
    left_clear = not np.any(left_band==0) and has_clear_gap(left_band)
    right_clear= not np.any(right_band==0) and has_clear_gap(right_band)

    if zero_ratio>=0.6:
        if   left_clear:  dec="Turning Left"
        elif right_clear: dec="Turning Right"
        else:             dec="Stopping + Handbrake"
    elif zero_ratio>=0.3:
        dec="Slowing Down"
    elif not vertical_clear:
        dec="Stopping"
    elif not (detect_obj_count>150):
        dec="Moving Forward"

    cv.putText(vis_overlay, f"Decision: {dec}", (30,30),
               cv.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    cv.putText(vis_overlay, f"Object Count: {detect_obj_count}", (30,60),
               cv.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    text_sz = cv.getTextSize(f"Road: {road}", cv.FONT_HERSHEY_SIMPLEX,0.8,2)[0]
    cv.putText(vis_overlay, f"Road: {road}", (w-text_sz[0]-10,30),
               cv.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

    cv.imshow("Navigation Debug View", vis_overlay)
    cv.imshow("Segmented", segmented)
    cv.imshow("Road Mask", road_mask)
    cv.imshow("Final Road Overlay", overlay)

    if cv.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv.destroyAllWindows()
