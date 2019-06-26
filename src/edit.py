# from bs4 import BeautifulSoup
# import re
# import os
# from svglib.svglib import svg2rlg
# from reportlab.graphics import renderPDF, renderPM



# # def convert_png(self, filename):
# #     """ Write the svg named filename.svg to a png file of the same name"""
# #     if filename[-3:] != "svg":
# #         print("filename is not svg")
# #         return
# #     fname = filename[:-4]
# #     drawing = svg2rlg(filename)
# #     renderPM.drawToFile(drawing, fname + ".png", fmt="PNG")


# def main():
#     d = "/Users/home1/Allen/summer19/src/"

#     out_d = "~/Allen/summer19/editedpolygon/"

#     for fname in os.listdir(d):

#         if fname[-3:] == "svg":

#             with open(fname, "r+") as f:

#                 os.rename(fname, fname[:-3] + "html")


#     for fname in os.listdir(d):

#         if fname[-4:] == "html":

#             with open(fname, "r+") as f:

#                 new_f = f.read()

#                 # select all the path tags
#                 soup = BeautifulSoup(new_f)
#                 paths = soup.find_all('path')

#                 print(paths[2])

#                 for p in paths[50:54]:
#                     # select the fill attribute
#                     style = p["style"]

#                     _, _, fill_style = style.partition(": ")

                    
                    
#                     # if white, leave as white
#                     if fill_style == "white":
#                         continue
#                     else:
#                         # assuming rgb(x, y, z)

#                         reg = re.compile("\d+, \d+, \d+")

#                         rgb = fill_style.match(reg)

#                         print(rgb)

#                         # else if rgb
#                         # grayscale rgb = 0.3 * R + 0.59 * G + 0.11 * B
#                         r, g, b = rgb.split(",")


#                     # save to svg

#                     # convert the resultant svg to png and save somewhere else


import os, cv2
import os.path



def main():

    basepath = "/Users/home1/Allen/summer19/"
    d = "/Users/home1/Allen/summer19/polygon/"

    savepath =  "/Users/home1/Allen/summer19/gray_polygon/"

    i = 0
    for fname in os.listdir(d):
        name = fname[:-4]
        ext = fname[-4:]
        if ext == ".png":
            img = cv2.imread(os.path.join(d, fname))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(savepath, f'gray_polygon{i}.png'), gray)
            i += 1





if __name__ == "__main__":
    main()