//
//  ViewController.swift
//  CoreML in ARKit
//
//  Created by Hanley Weng on 14/7/17.
//  Copyright © 2017 CompanyName. All rights reserved.
//

//MARK: Imports
import UIKit
import SceneKit
import ARKit
import AVFoundation
import CoreMedia
import Vision

//MARK: SCNIdentifiedObject Class this is used to store a SCNNode which has a label property for being displayed in the wold and having its text referrenced by other objects and methods
class SCNIdentifiedObject: SCNNode {
    var objectName: NSString?
    init(objectName: NSString) {
        super.init()
        self.objectName = objectName
    }
    
    /// Encode the identified object node.
    ///
    /// - Parameter aCoder: the encoder
    override func encode(with aCoder: NSCoder) {
        super.encode(with: aCoder)
        aCoder.encode(objectName, forKey: "objectName")
    }
    
    required init?(coder aDecoder: NSCoder) {
        super.init(coder: aDecoder)
        objectName = aDecoder.decodeObject(of: NSString.self, forKey: "objectName")
    }
}

//MARK: View Controller
class ViewController: UIViewController, ARSCNViewDelegate {
    
    //Create an instance of AVSpeechSynthesizer.
    var speechSynthesizer = AVSpeechSynthesizer()
    
    //variable describes whether or not to keep fixed object points or to attempt to live update the position of the objects in the scene.
    var updatePosition = true
    
    //Dictates the radius in which nodes will be combined distance in in meteres. I.E. Nodes within this radius with the same tag are assumed to be talking about the same object
    let recombiningThreshold: Double = 0.25
    
    // How many predictions we can do concurrently.
    static let maxInflightBuffers = 3
    
    //creates an instance of the functions that controll the bahavior of the YOLO nural network
    let yolo = YOLO()
    
    //MARK: Private variables
    var requests = [VNCoreMLRequest]()
    var startTimes: [CFTimeInterval] = []

    //creates an instance of a structure which can store a static image for processing
    let ciContext = CIContext()
    var resizedPixelBuffers: [CVPixelBuffer?] = []
    
    var framesDone = 0
    var frameCapturingStartTime = CACurrentMediaTime()
    
    var inflightBuffer = 0
    
    //the distance from the user in Metres that objects will be announced from if the user presses the button
    let queryRange = 1.5
    
    //can be either "crop" (lower area of effect) to perform processing on a selection of the area from the center of the screen or "scaleFit" to scale the whole image into the proper size and perform image processing on the whole image (lower resoulution)
    let imageProcessingSetting = "scaleFit"
    
    
    //MARK: Outlet
    
    // SCENE
    
    @IBOutlet weak var updatePositionsSwitch: UISwitch!
    
    //toggles whether or not the points will have their positions updated or not
    @IBAction func updatePositionsToggle(_ sender: Any) {
        if updatePosition == true {
            updatePosition = false
        }else {
            updatePosition = true
        }
    }
    
    @IBOutlet var sceneView: ARSCNView!
    let bubbleDepth : Float = 0.01 // the 'depth' of 3D text
    var latestPrediction : String = "…" // a variable containing the latest CoreML prediction
    
    @IBOutlet weak var debugImageView: UIImageView!
    
    @IBOutlet weak var queryEnvButton: UIButton!
    
    @IBAction func queryEnvironment(_ sender: Any) {
        //iterate throguh the nodes in the scene
        for child:SCNNode in sceneView.scene.rootNode.childNodes{
            
            if let child = child as? SCNIdentifiedObject, let pov = sceneView.pointOfView {
                
                //if the node is close enough to the user
                if d3Distance(pov.position, child.position) <= Double(queryRange){
                    // Creates an instance of AVSpeechUtterance and pass in a String to be spoken. This creates a synthesis of the string as though it is being spoken
                    let speechUtterance: AVSpeechUtterance = AVSpeechUtterance(string: "\(child.objectName!) accurate. Distance: \(round(10*d3Distance(pov.position, child.position))/10) meters.")
                    //Specify the speech utterance rate. The higher the values the slower speech patterns. The default rate, AVSpeechUtteranceDefaultSpeechRate is 0.5
                    speechUtterance.rate = 0.5
                    // Line 5. Pass in the urrerance to the synthesizer to actually speak.
                    speechSynthesizer.speak(speechUtterance)
                }
            }
        }
    }
    
    // COREML
    // A Serial Queue used in multithreading
    let dispatchQueueML = DispatchQueue(label: "com.hw.dispatchqueueml")
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Set the view's delegate
        sceneView.delegate = self
        
        // Show statistics such as fps and timing information
        sceneView.showsStatistics = true
        
        // Create a new scene
        let scene = SCNScene()
        
        // Set the scene to the view
        sceneView.scene = scene
        
        // Enable Default Lighting - makes the 3D text a bit more visible.
        sceneView.autoenablesDefaultLighting = true
        
        //////////////////////////////////////////////////
        
        //calls initializer functions
        setUpCoreImage()
        setUpVision()
        
        // Begin Loop to Update CoreML
        loopCoreMLUpdate()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        // Create an ARSession configuration
        let configuration = ARWorldTrackingConfiguration()
        // Enable plane detection
        configuration.planeDetection = [.horizontal,.vertical]
        
        // Run the view's session
        sceneView.session.run(configuration)
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        // Pause the view's session
        sceneView.session.pause()
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        print(#function)
    }
    
    // MARK: - Initialization
    func setUpCoreImage() {
        // Since we might be running several requests in parallel, we also need
        // to do the resizing in different pixel buffers or we might overwrite a
        // pixel buffer that's already in use.
        // one pixel buffer will hold one image section which we are going to be checking. thius one pixel buffer is an 'image' or image section representing one object
        for _ in 0..<YOLO.maxBoundingBoxes {
            //CVPixel Buffer is an image stored as pixels
            var emptyPixelBuffer: CVPixelBuffer?
            let status = CVPixelBufferCreate(nil, YOLO.inputWidth, YOLO.inputHeight,
                                             kCVPixelFormatType_32BGRA, nil,
                                             &emptyPixelBuffer)
            //error handeling
            if status != kCVReturnSuccess {
                print("Error: could not create resized pixel buffer", status)
            }
            //add the pixel buffer/image
            resizedPixelBuffers.append(emptyPixelBuffer)
        }
    }
    
    func setUpVision() {
        //attempts to create the yolo neural network
        guard let selectedModel = try? VNCoreMLModel(for: yolo.model.model) else {
            print("Error: could not create Vision model")
            return
        }
        
        for _ in 0..<ViewController.maxInflightBuffers {
            //sets up a bunch of requests to send to the neural network to make sure that it is not overloaded and we do not try to request more objects than the network can actually provide
            let request = VNCoreMLRequest(model: selectedModel, completionHandler: visionRequestDidComplete)
            
            requests.append(request)
        }
    }
    
    // MARK: - ARSCNViewDelegate
    
    func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
        DispatchQueue.main.async {
            // Do any desired updates to SceneKit here.
        }
    }
    
    // MARK: - UI stuff
    
    //arrainges views on the screen
    override func viewWillLayoutSubviews() {
        super.viewWillLayoutSubviews()
        resizePreviewLayer()
    }
    
    //sets the status bar style
    override var preferredStatusBarStyle: UIStatusBarStyle {
        return .lightContent
    }
    
    //sets the video from the camera to take up all of its allocated space
    func resizePreviewLayer() {
        sceneView.layer.frame = sceneView.bounds
    }
    
    
    //hides the status bar
    override var prefersStatusBarHidden : Bool {
        return true
    }
    
    // MARK: - Utility Functions
    
    //creates a new text node with the properly formatted text
    func createNewBubbleParentNode(_ text : String) -> SCNNode {
        // Warning: Creating 3D Text is susceptible to crashing. To reduce chances of crashing; reduce number of polygons, letters, smoothness, etc.
        
        // TEXT BILLBOARD CONSTRAINT
        let billboardConstraint = SCNBillboardConstraint()
        billboardConstraint.freeAxes = SCNBillboardAxis.Y
        
        // BUBBLE-TEXT
        let bubble = SCNText(string: text, extrusionDepth: CGFloat(bubbleDepth))
        let font = UIFont(name: "Futura", size: 0.15)
        bubble.font = font
        bubble.alignmentMode = kCAAlignmentCenter
        bubble.firstMaterial?.diffuse.contents = UIColor.orange
        bubble.firstMaterial?.specular.contents = UIColor.white
        bubble.firstMaterial?.isDoubleSided = true
        // bubble.flatness // setting this too low can cause crashes.
        bubble.chamferRadius = CGFloat(bubbleDepth)
        
        // BUBBLE NODE
        let (minBound, maxBound) = bubble.boundingBox
        let bubbleNode = SCNNode(geometry: bubble)
        // Centre Node - to Centre-Bottom point
        bubbleNode.pivot = SCNMatrix4MakeTranslation( (maxBound.x - minBound.x)/2, minBound.y, bubbleDepth/2)
        // Reduce default text size
        bubbleNode.scale = SCNVector3Make(0.2, 0.2, 0.2)
        
        // CENTRE POINT NODE
        let sphere = SCNSphere(radius: 0.005)
        sphere.firstMaterial?.diffuse.contents = UIColor.cyan
        let sphereNode = SCNNode(geometry: sphere)
        
        // BUBBLE PARENT NODE
        let bubbleNodeParent = SCNIdentifiedObject(objectName: NSString(string: text))
        bubbleNodeParent.addChildNode(bubbleNode)
        bubbleNodeParent.addChildNode(sphereNode)
        bubbleNodeParent.constraints = [billboardConstraint]
        
        return bubbleNodeParent
    }
    
    //Add a 3d marker at the given location with the given label
    func add3dLabel(label: String, certanty : Float, point : CGPoint, updatePosition: Bool){
        //performs a hit test to find the closest point in real world space to the projected ray from the inputted screen location
        let arHitTestResults : [ARHitTestResult] = sceneView.hitTest(CGPoint(x: point.x,y: point.y), types: [.existingPlaneUsingGeometry]) // Alternatively, we could use '.existingPlaneUsingExtent' for more grounded hit-test-points.
        if let closestResult = arHitTestResults.first {
            
            // Get Coordinates of the neares hit point in world space
            let transform : matrix_float4x4 = closestResult.worldTransform
            let worldCoord : SCNVector3 = SCNVector3Make(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
            
            //variables holding the sum of the positions of the nearby nodes and the number of nodes referencing the same object. these are used for calculating averages
            var nodePositionSum: SCNVector3 = worldCoord
            var nodeNumber: Int? = 1
            
            // Create 3D Text
            for child:SCNNode in sceneView.scene.rootNode.childNodes{
                //if there is a node in the vacinity with the same object type it is assumed that these two nodes refer to the same object
                if let child = child as? SCNIdentifiedObject, let objectName = child.objectName, d3Distance(child.position,worldCoord) <= recombiningThreshold && String(objectName).contains(label) {
                    //if we want to update the position of the object and place it at the average of all the points (this minimizes error due to ARKIT inaccuracies  and it can help account for movemnt of objects)
                    if updatePosition == true {
                        //increments the sum and total number of nodes used (useful for computing averages)
                        nodePositionSum = SCNVector3(nodePositionSum.x+child.position.x,nodePositionSum.y+child.position.y,nodePositionSum.z+child.position.z)
                        nodeNumber! += 1
                        
                        //removes the node so it does not display extra labels
                        child.removeFromParentNode()
                    //if we merely want to assume that the objects are fixed
                    }else{
                        //discard the curent saved location in favor of the past reading. by declairing a nil signifier
                        nodeNumber = nil
                    }
                }
            }
            
            //if we want to keep the past values of objects and we already had an object in the scene.
            if nodeNumber == nil {
                
                return
            //if we want to place a node in the scene or update the position of a node
            }else{
                
                //create a new node with the label of [Type Label]: [Percent Certanty rounded to the nearest tenth of a percentile]%
                let node : SCNNode = createNewBubbleParentNode("\(label): \(round(1000*certanty)/10)%")
                sceneView.scene.rootNode.addChildNode(node)
                //set the position to be equal to the average node position for new nodes this is just the location found for the node because this is the average of one value
                node.position = SCNVector3(nodePositionSum.x/Float(nodeNumber!),nodePositionSum.y/Float(nodeNumber!),nodePositionSum.z/Float(nodeNumber!))
                print("label \(label) certanty \(round(1000*certanty)/10) world \(worldCoord) actual \(node.position)")
            }
        }
    }
    
    //find the 3dDistance between two world points
    func d3Distance(_ point1 : SCNVector3, _ point2: SCNVector3) -> Double{
        return Double(sqrt(Double(pow((point1.x-point2.x),2))+Double(pow((point1.y-point2.y),2))+Double(pow((point1.z-point2.z),2))))
    }
    
    //displays a CVPixelBuffer to the image debugger
    func displayImage(buffer: CVPixelBuffer, debugImageView: UIImageView) -> Int{
        
        //convert the buffer to a CIImage
        let ciimage = CIImage(cvImageBuffer: buffer)
        
        //Convert into a UIImage
        let image = UIImage(ciImage: ciimage)
        
        //display the image
        debugImageView.image = image
        
        return 0
    }
    
    //calculate the cordinates for the a given point in the small image in the chordiantes of the view
    func calculatePointCords(yoloPoint: CGPoint, view: UIView, sceneView: ARSCNView, imageProcessingSetting: String) -> CGPoint? {
        
        //gets the dimenasions of the display view
        let viewShort = view.bounds.width
        let viewLong = view.bounds.height
        
        // Get the dimensions of the camera image a note is that the camera image is rotated 90 degrees conterclockwise from the phone screen when it is held in its vertical position.
        let imageLong = CVPixelBufferGetWidth((sceneView.session.currentFrame?.capturedImage)!)
        let imageShort = CVPixelBufferGetHeight((sceneView.session.currentFrame?.capturedImage)!)
        
        //get the size of the image which needs to be passed to the ML
        let YOLOSize = YOLO.inputHeight
        
        
        //changes the cropping/scaling options for getting the cordinates of the screen point based on the image that gets passed to the coreMl model
        switch imageProcessingSetting{
        case "crop":
            
            let imagePixelLocationShort = yoloPoint.x * CGFloat(YOLOSize) + CGFloat((imageShort-YOLOSize)/2)
            let imagePixelLocationLong = yoloPoint.y * CGFloat(YOLOSize) + CGFloat((imageLong-YOLOSize)/2)
            
            let viewPixelLocationShort = CGFloat(viewShort)/CGFloat(imageShort) * imagePixelLocationShort
            
            let viewPixelLocationLong = CGFloat(viewLong)/CGFloat(imageLong) * imagePixelLocationLong
            
            print("x \(viewPixelLocationShort), y \(viewPixelLocationLong)")
            
            return CGPoint(x: viewPixelLocationShort, y: viewPixelLocationLong)
        case "scaleFit":
            
            //scale the cordinates from the ML model to the respective cordinates in the screen space
            let viewPixelLocationShort = yoloPoint.x * CGFloat(YOLOSize) * CGFloat(viewShort/CGFloat(imageShort)*CGFloat(imageLong)/CGFloat(YOLOSize)) - CGFloat(20)
            let viewPixelLocationLong = yoloPoint.y * CGFloat(YOLOSize) * CGFloat(viewLong/CGFloat(imageLong)*CGFloat(imageLong)/CGFloat(YOLOSize))

            //return the scaled cordinates
            return CGPoint(x: viewPixelLocationShort, y: viewPixelLocationLong)
            
            
            
        default:
            print("passed improper scaling option to calculatePointCoords")
            return nil
        }
        
        
    }
    
    // MARK: - CoreML Vision Handling
    
    func loopCoreMLUpdate() {
        // Continuously run CoreML whenever it's ready. (Preventing 'hiccups' in Frame Rate)
        
        dispatchQueueML.async {
            // 1. Run Update.
            self.updateCoreML()
            
            // 2. Loop this function.
            self.loopCoreMLUpdate()
        }
        
    }

    func predict(pixelBuffer: CVPixelBuffer, inflightIndex: Int, imageProcessingSetting:String) {
        // Measure how long it takes to predict a single video frame.
        let startTime = CACurrentMediaTime()
        
        // Resize the input to the coreML Model with Core Image to 416x416.
        if let resizedPixelBuffer = resizedPixelBuffers[inflightIndex] {
            //converts the pixel buffer to a CI image
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            var orientedImage: CIImage?
            
            //changes the scaling/cropping option of the image sent into the core Ml Model
            switch imageProcessingSetting {
            case "crop":
                //crops the proper size out of the center of the image
                let croppedImage = ciImage.cropped(to: CGRect(x: (ciImage.extent.size.width/2)-CGFloat(YOLO.inputHeight/2), y: (ciImage.extent.size.height/2)-CGFloat(YOLO.inputWidth/2), width: CGFloat(YOLO.inputHeight), height: CGFloat(YOLO.inputWidth)))
                orientedImage = croppedImage.oriented(CGImagePropertyOrientation.right)
            case "scaleFit" :
                //scales the whole image to fit  so that the aspect ratio is maintained and the longest length is scaled to take up one side. this leads to the full image being captured however it also means that there is some empty space in the picture that the model processes
                let sy = CGFloat(YOLO.inputHeight) / CGFloat(ciImage.extent.size.width)
                let sx = sy
                let scaleTransform = CGAffineTransform(scaleX: sx, y: sy)
                let scaledImage = ciImage.transformed(by: scaleTransform)
                //rotates the image properly so that it is right side up from the model's perspective
                orientedImage = scaledImage.oriented(CGImagePropertyOrientation.right)

            default :
                orientedImage = nil
                print("improper imageProcessingSetting in Predict(pixelBuffer: ...")
                return
            }

            //converts to a pixelbuffer
            ciContext.render(orientedImage!, to: resizedPixelBuffer)
            //ciContext.render(ciImage, to: resizedPixelBuffer)
            
            // Give the resized input to our model.
            if let result = try? yolo.predict(image: resizedPixelBuffer){
                let elapsed = CACurrentMediaTime() - startTime
                showOnMainThread(result, elapsed)
            } else {
                //if the model could not find anything
                print("An error occurred and the model returned empty predictions")
            }
        }
    }
    
    func visionRequestDidComplete(request: VNRequest, error: Error?) {
        
    }
    
    func showOnMainThread(_ predictions: [YOLO.Prediction], _ elapsed: CFTimeInterval) {
        //redraw the bounding boxes
        DispatchQueue.main.async {
            //show the predictions in the view
            self.show(predictions: predictions)
        }
    }
    
    func show(predictions: [YOLO.Prediction]) {
        //iterate through all of the bounding boxes
        for i in 0..<predictions.count {
                let prediction = predictions[i]
            
            //corrects the origin of the bounding box which was given by YOLO to be in the cordinates of the view space
            let origin = calculatePointCords(yoloPoint: CGPoint(x: prediction.rect.origin.x, y: prediction.rect.origin.y), view: view, sceneView: sceneView, imageProcessingSetting: imageProcessingSetting)
            
            //if the height of the bounding box is taller than the width then use a bottom projection else use center projection
            if prediction.rect.height <= prediction.rect.width {
                
                //Adds a 3d label to the point at the origina of the bounding box.
                add3dLabel(label: String(labels[prediction.classIndex]), certanty: prediction.score, point: CGPoint(x: CGFloat(origin!.x), y: CGFloat(origin!.y)), updatePosition: updatePosition)
                
            }else{
                //Adds a 3d label to the point at the origina of the bounding box.
                add3dLabel(label: String(labels[prediction.classIndex]), certanty: prediction.score, point: CGPoint(x: CGFloat(origin!.x), y:( CGFloat(origin!.y)+CGFloat(prediction.rect.height/2))), updatePosition: updatePosition)
            }


        }
    }
    
    func updateCoreML() {
        // Get Camera Image as RGB
        let pixbuff : CVPixelBuffer? = (sceneView.session.currentFrame?.capturedImage)
        if pixbuff == nil { return }
        
        
        // For better throughput, we want to schedule multiple prediction requests
        // in parallel. These need to be separate instances, and inflightBuffer is
        // the index of the current request.
        let inflightIndex = inflightBuffer
        inflightBuffer += 1
        if inflightBuffer >= ViewController.maxInflightBuffers {
            inflightBuffer = 0
        }

        // For better throughput, perform the prediction on a concurrent
        // background queue instead of on the serial VideoCapture queue.
        DispatchQueue.global().async {
            //make a prediction about the contents of the image
            self.predict(pixelBuffer: pixbuff!, inflightIndex: inflightIndex, imageProcessingSetting: self.imageProcessingSetting)

        }
        //convert the current frae into a CI image
        let ciImage = CIImage(cvPixelBuffer: pixbuff!)

        // Prepare CoreML/Vision Request
        let imageRequestHandler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        // Run Image Request
        do {
            try imageRequestHandler.perform(self.requests)
        } catch {
            print(error)
        }
    }
}
