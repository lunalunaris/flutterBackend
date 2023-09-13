
class BoundBox:
    value = ""
    possibleValues=[]
    latexValue = ""
    binary = []
    projectionHorizontal = []  # 25 sized pixel count
    projectionVertical = []  # 25 sized pixel count
    zones = []  # 25 zones with pixel count
    crossingsVertical = []
    crossingsHorizontal = []  # vetical and horizontal count of groups of pixels
    merged=0
    featureArray=[]
    def __init__(self, height, width, coords, pixels):
        self.height = height
        self.width = width
        self.coords = coords
        self.pixels = pixels
