import PIL.Image
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, imshow, axis
import math
import os
import keras.models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = keras.models.load_model('model3.h5')

# IMAGE UPLOAD

def loadImage(filepath):
    img_orig = PIL.Image.open(filepath)
    img_width, img_height = img_orig.size
    aspect_ratio = min(560.0/img_width, 560.0/img_height)
    new_width, new_height = ((np.array(img_orig.size) * aspect_ratio)).astype(int)
    img = img_orig.resize((new_width,new_height), resample=PIL.Image.BILINEAR)
    img = img.convert('L')
    img = np.array(img)
    
    return img

# SADDLE POINTS

def getSaddle(gray_img):
    img = gray_img.astype(np.float64)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    gxx = cv2.Sobel(gx, cv2.CV_64F, 1, 0)
    gyy = cv2.Sobel(gy, cv2.CV_64F, 0, 1)
    gxy = cv2.Sobel(gx, cv2.CV_64F, 0, 1)
    S = gxx*gyy - gxy**2
    return S

def pruneSaddle(s):
    thresh = 128
    score = (s > 0).sum()
    while (score > 10000):
        thresh = thresh*2
        s[s < thresh] = 0
        score = (s > 0).sum()

def nonmax_sup(img, win = 10):
    w, h = img.shape
    img_sup = np.zeros_like(img, dtype=np.float64)
    for i, j in np.argwhere(img):
        a = max(0, i - win)
        b = min(w, i + win + 1)
        c = max(0, j - win)
        d = min(h, j + win + 1)
        cell = img[a:b, c:d]
        val = img[i, j]
        if cell.max() == val:
            img_sup[i, j] = val
    return img_sup

def getMinSaddleDist(saddle_pts, pt):
    best_dist = None
    best_pt = pt
    for saddle_pt in saddle_pts:
        saddle_pt = saddle_pt[::-1]
        dist = np.sum((saddle_pt - pt)**2)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_pt = saddle_pt
    return best_pt, np.sqrt(best_dist)

# CONTOURS

def simplifyContours(contours):
    for i in range(len(contours)):
        contours[i] = cv2.approxPolyDP(contours[i], 0.04*cv2.arcLength(contours[i], True), True)

def getContours(edges):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    edges_gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
    contours, hierarchy = cv2.findContours(edges_gradient, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    simplifyContours(contours)  
    return np.array(contours, dtype=object), hierarchy[0]

def getAngle(a, b, c):
    k = (a*a + b*b - c*c) / (2*a*b)
    if (k < -1):
        k = -1
    elif k > 1:
        k = 1
    return np.arccos(k) * 180.0 / np.pi

def is_square(cnt):

    dd0 = np.sqrt(((cnt[0,:] - cnt[1,:])**2).sum())
    dd1 = np.sqrt(((cnt[1,:] - cnt[2,:])**2).sum())
    dd2 = np.sqrt(((cnt[2,:] - cnt[3,:])**2).sum())
    dd3 = np.sqrt(((cnt[3,:] - cnt[0,:])**2).sum())
    xa = np.sqrt(((cnt[0,:] - cnt[2,:])**2).sum())
    xb = np.sqrt(((cnt[1,:] - cnt[3,:])**2).sum())

    ta = getAngle(dd3, dd0, xb) 
    tb = getAngle(dd0, dd1, xa)
    tc = getAngle(dd1, dd2, xb)
    td = getAngle(dd2, dd3, xa)

    angles = np.array([ta,tb,tc,td])
    good_angles = np.all((angles > 40) & (angles < 140))
    return good_angles

def updateCorners(contour, saddle, ws = 4):
    new_contour = contour.copy()
    for i in range(len(contour)):
        cc, rr = contour[i,0,:]
        rl = max(0,rr-ws)
        cl = max(0,cc-ws)
        window = saddle[rl:min(saddle.shape[0],rr+ws+1),cl:min(saddle.shape[1],cc+ws+1)]
        br, bc = np.unravel_index(window.argmax(), window.shape)
        s_score = window[br,bc]
        br -= min(ws,rl)
        bc -= min(ws,cl)
        if s_score > 0:
            new_contour[i,0,:] = cc+bc,rr+br
        else:
            return []
    return new_contour

def pruneContours(contours, hierarchy, saddle):
    new_contours = []
    new_hierarchies = []
    for i in range(len(contours)):
        cnt = contours[i]
        h = hierarchy[i]
        if h[2] != -1:
            continue
        if len(cnt) != 4:
            continue
        if cv2.contourArea(cnt) < 8*8:
            continue
        if not is_square(cnt):
            continue
        cnt = updateCorners(cnt, saddle)
        if len(cnt) != 4:
            continue
        new_contours.append(cnt)
        new_hierarchies.append(h)
    
    new_contours = np.array(new_contours)
    new_hierarchy = np.array(new_hierarchies)
    if len(new_contours) == 0:
        return new_contours, new_hierarchy
  
    areas = [cv2.contourArea(c) for c in new_contours]
    mask = [areas >= np.median(areas)*0.25] and [areas <= np.median(areas)*2.0]
    new_contours = new_contours[tuple(mask)]
    new_hierarchy = new_hierarchy[tuple(mask)]
    return np.array(new_contours), np.array(new_hierarchy)

# CHESS GRID
        
def getIdentityGrid(N):
    a = np.arange(N)
    b = a.copy()
    aa, bb = np.meshgrid(a, b)
    Igrid = np.vstack([aa.flatten(), bb.flatten()]).T
    return Igrid

def makeChessGrid(M, N=1):
    ideal_grid = getIdentityGrid(2+2*N)-N
    ideal_grid_pad = np.pad(ideal_grid, ((0,0),(0,1)), 'constant', constant_values=1)
    grid = (np.matrix(M)*ideal_grid_pad.T).T
    grid[:,:2] /= grid[:,2]
    grid = grid[:,:2]
    return grid, ideal_grid, M

def getInitChessGrid(quad):
    quadA = np.array([[0,1],[1,1],[1,0],[0,0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quadA, quad.astype(np.float32))
    return makeChessGrid(M)

def getChessGrid(quad):
    quadA = np.array([[0,1],[1,1],[1,0],[0,0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quadA, quad.astype(np.float32))
    quadB = getIdentityGrid(4) - 1
    quadB_pad = np.pad(quadB, ((0,0),(0,1)), 'constant', constant_values=1)
    C_grid = (np.matrix(M)*quadB_pad.T).T
    C_grid[:,:2] /= C_grid[:,2]
    return C_grid

def findGoodPoints(grid, spts, max_px_dist=5):
    new_grid = grid.copy()
    chosen_spts = set()
    N = len(new_grid)
    grid_good = np.zeros(N,dtype=bool)
    hash_pt = lambda pt: "%d_%d" % (pt[0], pt[1])
    
    for pt_i in range(N):
        pt2, d = getMinSaddleDist(spts, grid[pt_i,:2].A.flatten())
        if hash_pt(pt2) in chosen_spts:
            d = max_px_dist
        else:
            chosen_spts.add(hash_pt(pt2))
        if (d < max_px_dist):
            new_grid[pt_i,:2] = pt2
            grid_good[pt_i] = True
    return new_grid, grid_good

def generateNewBestFit(grid_ideal, grid, grid_good):
    a = np.float32(grid_ideal[grid_good])
    b = np.float32(grid[grid_good])
    M = cv2.findHomography(a, b, cv2.RANSAC)
    return M

def getBestLines(img_warped):
    img_warped = cv2.blur(img_warped,(5,5))
    gx = cv2.Sobel(img_warped,cv2.CV_64F,1,0)
    gy = cv2.Sobel(img_warped,cv2.CV_64F,0,1)

    gx_pos = gx.copy()
    gx_pos[gx_pos < 0] = 0
    gx_neg = -gx.copy()
    gx_neg[gx_neg < 0] = 0
    score_x = np.sum(gx_pos, axis=0) * np.sum(gx_neg, axis=0)

    gy_pos = gy.copy()
    gy_pos[gy_pos < 0] = 0
    gy_neg = -gy.copy()
    gy_neg[gy_neg < 0] = 0
    score_y = np.sum(gy_pos, axis=1) * np.sum(gy_neg, axis=1)
    
    a = np.array([(offset + np.arange(7) + 1)*32 for offset in np.arange(1,9)])
    scores_x = np.array([np.sum(score_x[pts]) for pts in a])
    scores_y = np.array([np.sum(score_y[pts]) for pts in a])
    
    best_lines_x = a[scores_x.argmax()]
    best_lines_y = a[scores_y.argmax()]
    return (best_lines_x, best_lines_y)

def findChessboard(img, min_pts_needed=15, max_pts_needed=25):
    blur_img = cv2.blur(img, (3,3))
    saddle = getSaddle(blur_img)
    saddle = -saddle
    saddle[saddle < 0] = 0
    pruneSaddle(saddle)
    s2 = nonmax_sup(saddle)
    s2[s2 < 100000] = 0
    spts = np.argwhere(s2)

    edges = cv2.Canny(img, 20, 250)
    contours_all, hierarchy = getContours(edges)
    contours, hierarchy = pruneContours(contours_all, hierarchy, saddle)
    
    curr_num_good = 0
    curr_grid_next = None
    curr_grid_good = None
    curr_M = None

    for cnt_i in range(len(contours)):
        cnt = contours[cnt_i].squeeze()
        grid_curr, ideal_grid, M = getInitChessGrid(cnt)

        for grid_i in range(7):
            grid_curr, ideal_grid, _ = makeChessGrid(M, N=(grid_i+1))
            grid_next, grid_good = findGoodPoints(grid_curr, spts)
            num_good = np.sum(grid_good)
            if num_good < 4:
                M = None
                break
            M, _ = generateNewBestFit(ideal_grid, grid_next, grid_good)
            if M is None or np.abs(M[0,0] / M[1,1]) > 15 or np.abs(M[1,1] / M[0,0]) > 15:
                M = None
                break
        if M is None:
            continue
        elif num_good > curr_num_good:
            curr_num_good = num_good
            curr_grid_next = grid_next
            curr_grid_good = grid_good
            curr_M = M

        if num_good > max_pts_needed:
            break
            
    if curr_num_good > min_pts_needed:
        final_ideal_grid = getIdentityGrid(2+2*7)-7
        return curr_M, final_ideal_grid, curr_grid_next, curr_grid_good, spts
    else:
        return None, None, None, None, None

def getUnwarpedPoints(best_lines_x, best_lines_y, M):
    x,y = np.meshgrid(best_lines_x, best_lines_y)
    xy = np.vstack([x.flatten(), y.flatten()]).T.astype(np.float32)
    xy = np.expand_dims(xy,0)

    xy_unwarp = cv2.perspectiveTransform(xy, M)
    return xy_unwarp[0,:,:]

def getBoardOutline(best_lines_x, best_lines_y, M):
    d = best_lines_x[1] - best_lines_x[0]
    ax = [best_lines_x[0]-d, best_lines_x[-1]+d]
    ay = [best_lines_y[0]-d, best_lines_y[-1]+d]
    x,y = np.meshgrid(ax, ay)
    xy = np.vstack([x.flatten(), y.flatten()]).T.astype(np.float32)
    xy = xy[[0,1,3,2,0],:]
    xy = np.expand_dims(xy,0)

    xy_unwarp = cv2.perspectiveTransform(xy, M)
    return xy_unwarp[0,:,:]

def image_transform(img, points, square_length=125):
	board_length = square_length * 8
	def __dis(a, b): return np.linalg.norm(np.array(a)-np.array(b))
	def __shi(seq, n=0): return seq[-(n % len(seq)):] + seq[:-(n % len(seq))]
	best_idx, best_val = 0, 10**6
	for idx, val in enumerate(points):
		val = __dis(val, [0, 0])
		if val < best_val:
			best_idx, best_val = idx, val
	pts1 = np.float32(__shi(points, 4 - best_idx))
	pts2 = np.float32([[0, 0], [board_length, 0], \
			[board_length, board_length], [0, board_length]])
	M = cv2.getPerspectiveTransform(pts1, pts2)
	W = cv2.warpPerspective(img, M, (board_length, board_length))
	return W

def llr_polysort(pts):
    mlat = sum(x[0] for x in pts) / len(pts)
    mlng = sum(x[1] for x in pts) / len(pts)
    def __sort(x):
        return (math.atan2(x[0]-mlat, x[1]-mlng) + 2*math.pi)%(2*math.pi)
    pts.sort(key=__sort)
    return pts

def decomposeFen(fen):
    FEN = ''
    fen = fen.replace('-', '')
    for i in fen:
        if i.isdigit():
            for _ in range(int(i)):
                FEN += 'o'
        else:
            FEN += i
    return FEN

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center == None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

mapping = {0:'bb', 1:'bk', 2:'bn', 3:'bp', 4:'bq', 5:'br', 6:'-', 7:'wb', 8:'wk', 9:'wn', 10:'wp', 11:'wq', 12:'wr'}

def produceFen(orig, mapping, model):
    fen = ''
    tiles = [orig[x:x+75,y:y+75] for x in range(0,orig.shape[0],75) for y in range(0,orig.shape[1],75)]
    for i in range(len(tiles)):
        tile = tiles[i]
        tile = cv2.cvtColor(tile,cv2.COLOR_GRAY2RGB)
        tile = cv2.resize(tile, (75,75))
        tile = tile.reshape(-1, 75, 75, 3)
        tile = tile/255.
        pred = model.predict(tile)
        indexs = np.argmax(pred[0])
        pred = list(map(lambda x: int(x*100), pred[0]))
        # print(i, pred)
        if i % 8 == 0:
            fen += '/'
        if mapping[indexs][0] == 'b':
            fen += mapping[indexs][1]
        if mapping[indexs][0] == 'w':
            fen += mapping[indexs][1].upper()
        if mapping[indexs] == '-':
            fen += mapping[indexs]

    fen = fen.replace('--------', '8')
    fen = fen.replace('-------', '7')
    fen = fen.replace('------', '6')
    fen = fen.replace('-----', '5')
    fen = fen.replace('----', '4')
    fen = fen.replace('---', '3')
    fen = fen.replace('--', '2')
    fen = fen.replace('-', '1')
    return fen[1:]

def correctOrientation(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    _, img_binary = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite('test.png', img_binary)
    colormap = []
    tiles = [img_binary[x:x+75,y:y+75] for x in range(0,img_binary.shape[0],75) for y in range(0,img_binary.shape[1],75)]
    for i in tiles:
        colormap.append(np.mean(i.flatten()))
    blackTile = np.argmax(colormap)
    if (blackTile + blackTile//8) % 2 == 0:
        return True
    return False

def buildPieceDataset():
    filenames = glob.glob(r'my\*.JPG')
    counter = 0
    for i in filenames:
        img = loadImage(i)
        img = cv2.resize(img, (600, 600))
        tiles = [img[x:x+75,y:y+75] for x in range(0,img.shape[0],75) for y in range(0,img.shape[1],75)]
        fen = i.split('\\')[-1][:-4]
        fen = decomposeFen(fen)
        print(fen)
        for j in range(len(tiles)):
            if fen[j].isupper() and fen[j] != 'o':
                pathing = r'figures\white\{}\{}.jpg'.format(fen[j].lower(), counter)
                cv2.imwrite(pathing, tiles[j])
                counter += 1
                continue
            if fen[j] != 'o':
                pathing = r'figures\black\{}\{}.jpg'.format(fen[j].lower(), counter)
                cv2.imwrite(pathing, tiles[j])
                counter += 1
                continue
            pathing = r'figures\blank\{}.jpg'.format(counter)
            cv2.imwrite(pathing, tiles[j])
            counter += 1

def testRecognition():
    filenames = glob.glob(r'chess-dataset\labeled_originals\*.JPG')
    for i in filenames:
        fen = i.split('\\')[-1]
        myImg, check = main(i)
        print('Orientation ' + str(check))
        if check:
            cv2.imwrite(r'chess-dataset\my\{}'.format(fen), myImg)

def main(filepath):
    print ("Processing %s" % (filepath))
    img = loadImage(filepath)
    M, ideal_grid, grid_next, grid_good, spts = findChessboard(img)

    if M is not None:
        M, _ = generateNewBestFit((ideal_grid+8)*32, grid_next, grid_good)
        img_warp = cv2.warpPerspective(img, M, (600, 600), flags=cv2.WARP_INVERSE_MAP)
        best_lines_x, best_lines_y = getBestLines(img_warp)
        board_outline_unwarp = getBoardOutline(best_lines_x, best_lines_y, M)
        four_points = [ele for ind, ele in enumerate(board_outline_unwarp) if ele not in board_outline_unwarp[:ind]]
        four_points = llr_polysort(four_points)

        img_warp_save = image_transform(img, four_points, 75)
        img_warp_save = cv2.flip(img_warp_save, 1)
        img_warp_save = rotate(img_warp_save, 90)
        if not correctOrientation(img_warp_save):
            img_warp_save = rotate(img_warp_save, 90)
        return produceFen(img_warp_save, mapping, model)

    print("Failed to capture chessboard")
    return None