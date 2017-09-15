from PIL import Image
import sys
im = Image.open(sys.argv[1])
print("load ",sys.argv[1])
width,height = im.size
pixel = im.load()
for y in range(0, height):
    for x in range(0, width):
        R, G, B = pixel[x,y]
        pixel[x,y] = (R//2,G//2,B//2)
im.save("Q2.png")