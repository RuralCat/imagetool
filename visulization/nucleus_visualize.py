import skimage.io as skio
import matplotlib.pyplot as plt
from matplotlib import lines

def plot_nucleus_bound(im, verts):
    # read image
    if isinstance(im, str): im = skio.imread(im)
    # create axes
    _, ax = plt.subplots(1, 1)
    ax.axis('off')
    # plot lines
    for i in range(len(verts)):
        l = lines.Line2D(verts[i][:,0], verts[i][:,1])
        ax.add_line(l)
    # show image
    plt.imshow(im)
    plt.show()


def plot_image(im):
    plt.imshow(im, cmap=plt.get_cmap('gray'))
    plt.axis('off')
    plt.show()

def multi_plot_image(imgs):
    plt.figure()
    num = len(imgs)
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.imshow(imgs[i], cmap=plt.get_cmap('gray'))
        plt.axis('off')
    plt.show()

def image_slice_show():
    import vtk

    # read image
    from PIL import Image
    import numpy as np
    im = Image.open('K:\BIGCAT\Projects\EM\data\Other data\EM CA1 hippocampus region of brain\\training.tif')
    imarray = np.array(im) / 255
    im.close()


    filename = "writeImageData.vti"

    imageData = vtk.vtkImageData()
    imageData.SetDimensions(imarray.shape[0], imarray.shape[1], 5)
    if vtk.VTK_MAJOR_VERSION <= 5:
        imageData.SetNumberOfScalarComponents(1)
        imageData.SetScalarTypeToDouble()
    else:
        imageData.AllocateScalars(vtk.VTK_DOUBLE, 1)

    dims = imageData.GetDimensions()

    # Fill every entry of the image data with "2.0"
    for z in range(dims[2]):
        for y in range(dims[1]):
            for x in range(dims[0]):
                imageData.SetScalarComponentFromDouble(x, y, z, 0, imarray[x, y])

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInputConnection(imageData.GetProducerPort())
    else:
        writer.SetInputData(imageData)
    writer.Write()

    # Read the file (to test that it was written correctly)
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()

    # Convert the image to a polydata
    imageDataGeometryFilter = vtk.vtkImageDataGeometryFilter()
    imageDataGeometryFilter.SetInputConnection(reader.GetOutputPort())
    imageDataGeometryFilter.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(imageDataGeometryFilter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(3)

    # Setup rendering
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)
    renderer.ResetCamera()

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)

    renderWindowInteractor = vtk.vtkRenderWindowInteractor()

    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindowInteractor.Initialize()
    renderWindowInteractor.Start()

def pyqt_win(name=''):
    pass

if __name__ == '__main__':
    image_slice_show()



