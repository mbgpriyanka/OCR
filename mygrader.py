#python mygrader.py --image rollnumberonly.jpg
#this program reads the image and draws contours over the bubbles and the numbers

# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract"

def read_process_image(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(blurred, 10, 200)
	cv2.imshow("edged", edged)
	cv2.waitKey(0)
	im = edged.copy()
	return edged

def get_draw_contours(im,ims):
	cnts = cv2.findContours(im, cv2.RETR_EXTERNAL,
						   cv2.CHAIN_APPROX_SIMPLE)
	pcsdimage = ims.copy()
	print('bef imutils grabcontours cnts', len(cnts))
	cnts = imutils.grab_contours(cnts)
	print('after cnts', len(cnts))
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	cv2.drawContours(ims, cnts, -1, (0, 255, 0), 3)
	cv2.imshow("Contours", ims)
	cv2.waitKey()

	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.1 * peri, True)
		x, y, w, h = cv2.boundingRect(approx)
		im = cv2.rectangle(pcsdimage,(x,y),((x+w),(y+h)),(0,0,255),1)
		cv2.imshow("1-by-1", pcsdimage)
		cv2.waitKey()

	return
def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
					help="path to the input image")
	args = vars(ap.parse_args())
	image = cv2.imread(args["image"])
	ims = image.copy()
	cv2.imshow("Original", image)
	cv2.waitKey(0)
	edged = read_process_image(image)
	im = edged.copy()
	get_draw_contours(im,ims)

if __name__ == "__main__":
	main()


'''
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())


image = cv2.imread(args["image"])
ims = image.copy()
cv2.imshow("Original", image)
cv2.waitKey(0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 10, 200) # these values highlight only the roll number and the darkened bubbles
cv2.imshow("edged", edged)
cv2.waitKey(0)
im = edged.copy()
cnts = cv2.findContours(im, cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

print('bef imutils grabcontours cnts',len(cnts))
cnts = imutils.grab_contours(cnts)
print('after cnts',len(cnts))
#docCnt = None

thresh = cv2.threshold(im, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# find contours in the thresholded image, then initialize
# the list of contours that correspond to questions
cnts = cv2.findContours(im, cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print('after thresh cnts',len(cnts))
cv2.imshow("thresh", thresh)
cv2.waitKey()
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
cv2.drawContours(ims,cnts,-1,(0,255,0),3)
cv2.imshow("sthresh", ims)
cv2.waitKey()
coords =[]
pcsdimage = image.copy()
for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.1 * peri, True)
	x, y, w, h = cv2.boundingRect(approx)
	coords.append(c)
	#im = cv2.rectangle(pcsdimage,(x,y),((x+w),(y+h)),(0,255,0),1)
	#data = pytesseract.image_to_string(pcsdimage[x:x+w,y,y+h], lang='eng', config='--psm 6')
	#print(data)
	#cv2.imshow('draw', im)
	#cv2.waitKey()
#print(coords)
#cv2.imshow('part',im[coords[0]])
cv2.waitKey()



cv2.waitKey()



'''

