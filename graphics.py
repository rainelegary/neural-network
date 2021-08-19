tk = __import__("tkinter")  # import tkinter as tk

graphics_pointSize = 0.1

class WindowSet:
    def __init__(self, title, geometry):
        self.ownWindow = tk.Tk()
        self.ownCanvas = None
        self.title = title
        self.geometry = geometry
        self.width = geometry.split('x')[0]
        self.height = geometry.split('x')[1]

        self.initUI()

    def initUI(self):
        ownWindow = self.ownWindow
        ownWindow.title(self.title)
        ownWindow.geometry(self.geometry)

        ownCanvas = tk.Canvas(ownWindow, bg='#303030')
        ownCanvas.place(x=0, y=0, width=self.width, height=self.height)

        self.ownWindow = ownWindow
        self.ownCanvas = ownCanvas


def drawFrame(canvasObject, points=(), lines=(), triangles=(), pointSize=2):
    canvasObject.delete("all")
    for point in points:
        canvasObject.create_oval(point[0]-pointSize/2, point[1]-pointSize/2, point[0]+pointSize/2, point[1]+pointSize/2, fill='#F0F0F0', outline='#F0F0F0')

    for line in lines:
        canvasObject.create_line(line, fill='#F0F0F0')

    for triangle in triangles:
        canvasObject.create_polygon(triangle, fill='#F0F0F0')


def loopWindow(windowObject, points=(), lines=(), triangles=(), pointSize=1):
    window = windowObject.ownWindow
    canvas = windowObject.ownCanvas

    drawFrame(canvas, points=points, lines=lines, triangles=triangles, pointSize=pointSize)
    window.update()


def coordsToPixel(coordsGiven):
    xIn = coordsGiven[0]
    yIn = coordsGiven[1]

    xOut = xIn * 300 + 400
    yOut = yIn * -300 + 300

    return xOut, yOut

