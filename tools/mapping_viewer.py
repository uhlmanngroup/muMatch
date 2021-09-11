import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from vedo import *
import copy


class MouseInteractorPlacePoint(vtk.vtkInteractorStyleTrackballCamera):
    '''
    DOTO.
    '''
    def __init__(self,_sphere1, _sphere2, _render1, _render2, _data1, _data2):

        self.AddObserver("LeftButtonPressEvent",self.leftButtonPressEvent)
        self.sphere1 = _sphere1
        self.sphere2 = _sphere2
        self.render1 = _render1
        self.render2 = _render2
        self.data1   = _data1
        self.data2   = _data2

        self.render2.SetActiveCamera( self.render1.GetActiveCamera())

        self.NumberOfClicks = 0
        self.ResetPixelDistance = 5
        self.PreviousPosition = [0,0]

        return


    def nearestVertex(self,pt,data):
        d = np.linalg.norm(data-pt, axis = -1)
        ind = np.argmin(d)
        return ind


    def clickDistance(self,clickPosition):
        dx = clickPosition[0] - self.PreviousPosition[0]
        dy = clickPosition[1] - self.PreviousPosition[1]
        moveDistance = np.sqrt(dx**2 + dy**2)
        return moveDistance


    def placePoint(self, clickPos):
        renderer = self.GetInteractor().FindPokedRenderer(*clickPos)
        picker = vtk.vtkCellPicker()
        picker.Pick(clickPos[0], clickPos[1], 0, renderer)
        worldPosition = picker.GetPickPosition()

        if (picker.GetCellId() != -1):
            if (renderer == self.render1):
                ind = self.nearestVertex(worldPosition,self.data1)
                self.sphere1.SetCenter(*self.data1[ind])
                self.sphere2.SetCenter(*self.data2[ind])
            else:
                ind = self.nearestVertex(worldPosition,self.data2)
                self.sphere1.SetCenter(*self.data1[ind])
                self.sphere2.SetCenter(*self.data2[ind])
        return


    def leftButtonPressEvent(self,obj,event):
        self.NumberOfClicks += 1
        clickPos = self.GetInteractor().GetEventPosition()
        moveDistance = self.clickDistance(clickPos)
        self.PreviousPosition = copy.deepcopy(clickPos)

        if(self.ResetPixelDistance < moveDistance):
            self.NumberOfClicks = 1

        if(self.NumberOfClicks == 2):
            self.NumberOfClicks = 0
            self.placePoint(clickPos)

        self.OnLeftButtonDown()
        return



def sphereActor(radius = 0.02):
    sphere = vtk.vtkSphereSource()
    sphere.SetPhiResolution(11)
    sphere.SetThetaResolution(21)

    sphere.SetRadius(radius)
    sphere.SetCenter(0,0,0)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetDiffuseColor(0,1,0)
    actor.GetProperty().SetDiffuse(.8)
    actor.GetProperty().SetSpecular(.5)
    actor.GetProperty().SetSpecularColor(1.0,1.0,1.0)
    actor.GetProperty().SetSpecularPower(30.0)

    return actor,sphere



def renderCorrespondence_vtk(actor1, actor2, idx, rad = 0.03):
    '''
    DOTO.
    '''

    actor1 = actor1.clone().normalize()
    actor2 = actor2.clone().normalize()

    points1 = actor1.points()
    points2 = actor2.points()
    #print(points1.shape, points2.shape, points2[idx].shape)

    sphereActor_1,sphere_1 = sphereActor(rad)
    sphereActor_2,sphere_2 = sphereActor(rad)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(2000, 1000)

    # Interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow);

    leftRenderer = vtk.vtkRenderer()
    renderWindow.AddRenderer(leftRenderer)
    leftRenderer.SetViewport([.0,.0,.5, 1.0])
    leftRenderer.SetBackground(.6, .5, .4)

    rightRenderer = vtk.vtkRenderer()
    renderWindow.AddRenderer(rightRenderer)
    rightRenderer.SetViewport([.5,.0,1.0,1.0])
    rightRenderer.SetBackground(.4, .5, .6)

    leftRenderer.AddActor(sphereActor_1)
    leftRenderer.AddActor(actor1)
    rightRenderer.AddActor(sphereActor_2)
    rightRenderer.AddActor(actor2)

    leftRenderer.ResetCamera()
    rightRenderer.ResetCamera()
    rightRenderer.GetActiveCamera().Azimuth(30)
    rightRenderer.GetActiveCamera().Elevation(30)

    style = MouseInteractorPlacePoint(sphere_1, sphere_2,
                                      leftRenderer, rightRenderer,
                                      points1, points2[idx])

    interactor.SetInteractorStyle(style)
    interactor.SetRenderWindow(renderWindow)


    renderWindow.Render()
    interactor.Start()




if __name__ == '__main__':
    pass
