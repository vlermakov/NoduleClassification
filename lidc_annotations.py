import xml.etree.ElementTree as ET
import math

ns = {'lidc': 'http://www.nih.gov'}

XML_ANNOTATIONS_DIRECTORY = "./LIDC-XML-only/"
MAX_READINGS_PER_SCAN = 4

def remove_whitespace(node):
    if node.nodeType == node.TEXT_NODE:
         if node.nodeValue.strip() == "":
             node.nodeValue = ""
    for child in node.childNodes:
         remove_whitespace(child)

class NoduleSlice:
    def __init__(self) -> None:
        self.z = None
        self.inclusion = True
        self.path = []

class Nodule:
    def __init__(self) -> None:
        self.noduleID = None
        self.positionZ = None
        self.positionX = None
        self.positionY = None
        self.diameter = None

        self.slices = []

    def estimateCenterAndDiameter(self):
        if(len(self.slices) > 0):
            self.positionZ = self.slices[0].z
            self.positionX = 0
            self.positionY = 0
            nPoints = 0
            for s in self.slices:
                for p in s.path:
                    self.positionX += p[0]
                    self.positionY += p[1]
                    nPoints = nPoints + 1
                
            self.positionX /= nPoints
            self.positionY /= nPoints

            self.radius = 0
            for s in self.slices:
                for p in s.path:
                    d = math.sqrt((p[0]-self.positionX)**2 + (p[1]-self.positionY)**2)
                    if(d > self.radius):
                        self.radius = d

class NonNodule(Nodule):
    def __init__(self) -> None:
        super().__init__()

class LargeNodule(Nodule):
    def __init__(self) -> None:
        super().__init__()
        self.subtlety = None
        self.internalStructure = None
        self.calcification = None
        self.sphericity = None
        self.margin = None
        self.lobulation = None
        self.spiculation = None
        self.texture = None
        self.malignancy = None

class ScanAnnotations:
    annotationFileIndex = {}

    def GetAnnotationsFile(self,scanId):
        # scan through all of the xml files in the 'tcia-lidc-xml'
        #  directory  and its subdirectories and check which has the file that contains scanId

        # Check if we have already found the file for this scanId
        if(scanId in self.annotationFileIndex):
            return self.annotationFileIndex[scanId]

        # Create a list of all xml files in the directory and subdirectories
        import os
        xmlFiles = []
        for root, dirs, files in os.walk(XML_ANNOTATIONS_DIRECTORY):
            for file in files:
                if file.endswith(".xml"):
                    xmlFiles.append(os.path.join(root, file))
        
        # Check each xml file to see if it contains the scanId
        for xmlFile in xmlFiles:
            try:
                tree = ET.parse(xmlFile)
                root = tree.getroot()
                header = root.find('lidc:ResponseHeader',ns)

                # Cache the file name for future use
                self.annotationFileIndex[header.find("lidc:SeriesInstanceUid",ns).text] = xmlFile
                
                if(header.find("lidc:SeriesInstanceUid",ns).text == scanId):
                    return xmlFile
            except:
                pass

        raise IOError("Could not find XML file for scan: {}".format(scanId))

    def __init__(self,scanId):
        self.Version = None
        self.MessageId = None
        self.DateRequest = None
        self.TimeRequest = None
        self.TaskDescription = None
        self.SeriesInstanceUid = None
        self.DateService = None
        self.TimeService = None
        self.ResponseDescription = None
        self.StudyInstanceUID = None

        self.readingSessions = []
        
        try:
            xmlFile = self.GetAnnotationsFile(scanId)
            print(f"Found annotations file {xmlFile}")
        except IOError:
            print("Error: Could not load XML file for scan: {}".format(scanId))
            exit()

        try:
            tree = ET.parse(xmlFile)
            root = tree.getroot()
            header = root.find('lidc:ResponseHeader',ns)
        except:
            raise IOError("Error: Could not parse XML file: {}".format(xmlFile))
        
        try:
            self.Version = header.find("lidc:Version",ns).text
            self.MessageId = header.find("lidc:MessageId",ns).text
            self.DateRequest = header.find("lidc:DateRequest",ns).text
            self.TimeRequest = header.find("lidc:TimeRequest",ns).text
            self.TaskDescription = header.find("lidc:TaskDescription",ns).text
            self.SeriesInstanceUid = header.find("lidc:SeriesInstanceUid",ns).text
            self.DateService = header.find("lidc:DateService",ns).text
            self.TimeService = header.find("lidc:TimeService",ns).text
            self.ResponseDescription = header.find("lidc:ResponseDescription",ns).text
            self.StudyInstanceUID = header.find("lidc:StudyInstanceUID",ns).text
        except:
            # These header parameters aren't very important
            pass

        self.readingSessions = []

        try:
            for iSession,session in enumerate(root.findall('lidc:readingSession',ns)):
                currentSession = []
                for nodule in session.findall('lidc:unblindedReadNodule',ns):
                    
                    characteristics = nodule.find('lidc:characteristics',ns)
                    n = None
                    
                    if(characteristics is not None):
                        n = LargeNodule()
                        n.noduleID = nodule.find('lidc:noduleID',ns).text
                        n.subtlety  = characteristics.find('lidc:subtlety',ns).text
                        n.internalStructure = characteristics.find('lidc:internalStructure',ns).text
                        n.calcification = characteristics.find('lidc:calcification',ns).text
                        n.sphericity = characteristics.find('lidc:sphericity',ns).text
                        n.margin = characteristics.find('lidc:margin',ns).text
                        n.lobulation = characteristics.find('lidc:lobulation',ns).text
                        n.spiculation = characteristics.find('lidc:spiculation',ns).text
                        n.texture = characteristics.find('lidc:texture',ns).text
                        n.malignancy = characteristics.find('lidc:malignancy',ns).text

                        for contour in nodule.findall('lidc:roi',ns):
                            slice = NoduleSlice()
                            slice.z = float(contour.find('lidc:imageZposition',ns).text)
                            slice.inclusion = (contour.find('lidc:inclusion',ns).text == "TRUE")
                            for point in contour.findall('lidc:edgeMap',ns):
                                slice.path.append((float(point.find('lidc:xCoord',ns).text),float(point.find('lidc:yCoord',ns).text)))
                            n.slices.append(slice)
                    else:
                        n = Nodule()
                        n.noduleID = nodule.find('lidc:noduleID',ns).text
                        for contour in nodule.findall('lidc:roi',ns):
                            slice = NoduleSlice()
                            slice.z = float(contour.find('lidc:imageZposition',ns).text)
                            slice.inclusion = contour.find('lidc:inclusion',ns).text
                            for point in contour.findall('lidc:edgeMap',ns):
                                slice.path.append((float(point.find('lidc:xCoord',ns).text),float(point.find('lidc:yCoord',ns).text)))
                            n.slices.append(slice)

                    n.estimateCenterAndDiameter()
                    currentSession.append(n)
                
                for nonNodule in session.findall('lidc:nonNodule',ns):
                    n = NonNodule()
                    n.noduleID = nonNodule.find('lidc:nonNoduleID',ns).text
                    for contour in nonNodule.findall('lidc:roi',ns):
                        slice = NoduleSlice()
                        slice.z = float(contour.find('lidc:imageZposition',ns).text)
                        slice.inclusion = contour.find('lidc:inclusion',ns).text
                        for point in contour.findall('lidc:edgeMap',ns):
                            slice.path.append((float(point.find('lidc:xCoord',ns).text),float(point.find('lidc:yCoord',ns).text)))
                        n.slices.append(slice)
                    n.estimateCenterAndDiameter()
                    currentSession.append(n)

                self.readingSessions.append(currentSession)
        except:
            raise IOError("Error: Could not parse XML file: {}".format(xmlFile))
        

