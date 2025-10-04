# Vision Transformer (ViT) on CIFAR-10 🖼️

## 🎯 Goal
The primary goal of this project was to **implement a Vision Transformer (ViT)** and train it on the **CIFAR-10 dataset**. The objective was to **achieve the highest possible test accuracy** by fine-tuning a pre-trained model, while also building a ViT from scratch to gain a deeper understanding of the architecture defined in the paper: **"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., ICLR 2021).**

This was implemented and trained entirely on **Google Colab**.

---

## 📌 Key Concepts

The implementation strictly follows the ViT pipeline:

1.  **Patchify Images**
    -   Each 32×32 image from CIFAR-10 is divided into smaller **non-overlapping patches**.
    -   This step transforms an image into a "sequence of tokens," making it suitable for a Transformer.

2.  **Learnable Positional Embeddings**
    -   To preserve **spatial relationships** between patches, trainable positional embeddings are added to the patch embeddings.

3.  **CLS Token (Classification Token)**
    -   A special **`[CLS]` token** is prepended to the sequence. The corresponding output from the transformer is used as the aggregate image representation for classification.

4.  **Transformer Encoder Blocks**
    -   The core of the model consists of stacked encoder blocks with **Multi-Head Self-Attention (MHSA)** and **MLP layers**, stabilized by residual connections and layer normalization.

5.  **Classification Head**
    -   The final `[CLS]` token embedding is passed through a simple MLP head to produce the final class probabilities.

---

## 🛠️ Implementation & Training Strategy

### Approach 1: Fine-Tuning a Pre-trained ViT for State-of-the-Art Performance
Achieving the highest possible accuracy necessitated a robust transfer learning strategy. The core principle was to leverage a large-scale, pre-trained Vision Transformer and adapt its powerful, generalized visual representations to the specific domain of the CIFAR-10 dataset. This approach is critical because Vision Transformers, lacking the inductive biases of CNNs, require enormous datasets to learn foundational visual features from scratch. Fine-tuning allows us to bypass this requirement and achieve superior performance.

-   **Model Backbone Selection (ViT-B/16):** The selected architecture was the **Vision Transformer Base/16 (`ViT-B/16`)**. The "Base" designation refers to a model with 12 transformer layers, a hidden embedding dimension of 768, and 12 self-attention heads. The "/16" indicates that the input image is decomposed into a grid of non-overlapping 16x16 pixel patches. This specific model, pre-trained on the vast ImageNet dataset, has already learned a rich hierarchy of visual features, from simple edges and textures to complex object components. This pre-existing knowledge is the foundation upon which our fine-tuning process is built.

-   **Framework and Model Sourcing (PyTorch and `timm`):** The implementation was built using **PyTorch**, a leading deep learning framework that provides the necessary flexibility for custom training loops and gradient management. The pre-trained `ViT-B/16` model was sourced via the **`timm` (PyTorch Image Models) library**, which is an essential tool for computer vision practitioners. `timm` provides a standardized, easy-to-use interface for accessing a vast collection of state-of-the-art model architectures and their corresponding pre-trained weights, significantly streamlining the development process.

-   **Advanced Data Augmentation:** To mitigate overfitting on the relatively small CIFAR-10 dataset, a multi-faceted data augmentation strategy was employed.
    -   **`TrivialAugmentWide`:** This is an advanced, automated augmentation policy. Instead of manually curating a sequence of transformations, `TrivialAugmentWide` randomly selects a single augmentation technique (e.g., shearing, color inversion, contrast adjustment) from a comprehensive set and applies it with a randomly sampled magnitude. This approach removes the need for expensive hyperparameter tuning of the augmentation pipeline and has been shown to yield state-of-the-art results.
    -   **Standard Geometric Augmentations:** `RandomCrop` and `HorizontalFlip` were included as fundamental transformations. Random cropping forces the model to learn features that are robust to changes in object position and scale, while horizontal flipping leverages the natural horizontal symmetry in most object classes to effectively increase the diversity of the training data.
    -   **Input Distribution Matching (ImageNet Normalization):** A critical step was normalizing the input CIFAR-10 images using the **mean and standard deviation statistics of the ImageNet dataset**. Since the `ViT-B/16` backbone was pre-trained on ImageNet, its weights are optimized for this specific input distribution. Applying the same normalization ensures that the data distribution during fine-tuning does not shift, which is essential for stable and effective knowledge transfer.

-   **Optimization and Learning Rate Scheduling:**
    -   **Optimizer (`AdamW`):** The **AdamW optimizer** was chosen over the standard Adam optimizer. `AdamW` decouples the weight decay term from the adaptive gradient updates. This separation corrects a subtle flaw in Adam where L2 regularization is not equivalent to true weight decay, leading to more effective regularization and often better model generalization.
    -   **Learning Rate Scheduler (`OneCycleLR`):** Training utilized the **OneCycleLR policy**, a highly effective learning rate scheduler. This policy operates in two phases: a warm-up phase where the learning rate linearly increases to a maximum value, followed by a cool-down phase where it anneals down. This method helps the model to traverse the loss landscape more effectively, enabling it to avoid sharp local minima and converge faster and to a better final solution.

-   **Advanced Regularization Techniques:**
    -   **`Label Smoothing`:** Instead of using one-hot encoded labels (which sets the target probability for the correct class to 1.0), **label smoothing** with a factor of `ε = 0.1` was applied. This technique reduces the target value for the correct class to `1 - ε` (0.9) and distributes `ε` across the other classes. This discourages the model from becoming overconfident in its predictions, leading to better calibration and improved generalization.
    -   **`Dropout`:** Standard **dropout layers** were active within the MLP blocks of the transformer. During training, dropout randomly sets a fraction of neuron activations to zero, preventing complex co-adaptations between neurons and forcing the model to learn more robust, redundant features. This acts as a form of model ensembling, improving overall performance.

### Approach 2: Building a ViT From Scratch
To understand the model's inner workings, a complete ViT was implemented from scratch in PyTorch. This model was composed of core modules like **Patch Embedding**, **Transformer Blocks**, and the final **Vision Transformer** class that integrates them. This approach allowed for experimentation with different architectural parameters.

---

## 📊 Results

### State-of-the-Art Accuracy with Fine-Tuning
Through the carefully designed training and regularization strategy described above, the fine-tuned `ViT-B/16` model achieved **state-of-the-art performance on CIFAR-10**.

| **Model** | **Test Accuracy (%)** |
| :--- | :--- |
| ResNet-18 (baseline CNN) | 94.5 |
| **Vision Transformer (ViT-B/16, Fine-Tuned)** | **✨ 98.7** |

This result significantly outperforms traditional CNN baselines, demonstrating the power of transfer learning with transformers for capturing global image dependencies.

### Insights from Scratch-Training Experiments
Training the ViT from scratch on CIFAR-10 yielded much lower accuracies, highlighting the architecture's dependence on large-scale pre-training.

| **Experiment** | Patch Size | Embed Dim | Layers | Heads | Best Validation Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1. Base Config | 4x4 | 128 | 6 | 4 | 25.03 |
| 2. Wider Model | 4x4 | 256 | 6 | 4 | 12.63 |
| 3. Deeper Model | 4x4 | 128 | 12 | 8 | 10.05 |
| 4. **Smaller Patches** | **2x2** | **128** | **6** | **4** | **🏆 25.33** |

---

## 🚀 Analysis & Key Takeaways

1.  **Transfer Learning is Paramount:** The enormous **~73% accuracy gap** between the fine-tuned model (98.7%) and the best from-scratch model (25.33%) is the most critical takeaway. Vision Transformers are **extremely data-hungry** and fail to generalize on small datasets like CIFAR-10 without prior learning on a massive dataset like ImageNet.

2.  **Adapting to Small Images and Datasets:** Even though CIFAR-10 images are small (32×32), applying a transformer was non-trivial. The success of the fine-tuned model depended on adapting the pre-trained positional embeddings and using strong regularization and data augmentation to prevent overfitting.

3.  **Depth/Width Trade-offs for Scratch Training:** The from-scratch experiments showed that increasing model capacity (making it wider or deeper) was **detrimental** to performance on CIFAR-10. This indicates a "less is more" approach is needed, as larger models overfit more severely without a large dataset to learn from.

4.  **Patch Size Matters:** Using a smaller `2x2` patch size (creating a longer sequence of 256 tokens) gave a slight edge over `4x4` patches (64 tokens) in the from-scratch experiments. This suggests that for low-resolution images, higher sequence granularity can be beneficial for capturing finer details.

---

## 📖 References
-   Dosovitskiy et al., 2021: *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* ([Paper Link](https://arxiv.org/abs/2010.11929))
-   PyTorch & Timm Documentation
-   CIFAR-10 Dataset ([Krizhevsky et al.](https://www.cs.toronto.edu/~kriz/cifar.html))





# Text-Prompted Image and Video Segmentation with GroundingDINO and SAM 2

## 🎯 Project Goal
The primary objective of this project is to implement a sophisticated, end-to-end pipeline for performing text-prompted segmentation on a single, static image. This involves translating a natural language description into precise pixel-level masks for the described objects.

As an extension, the project also demonstrates a more advanced capability: **text-driven video object segmentation**. This bonus objective involves identifying an object in a single frame based on a text prompt and then autonomously propagating its segmentation mask through the subsequent frames of a video clip, effectively creating a tracked object mask over time.

---

## 🔬 Core Technologies
This pipeline achieves its objective by synergistically combining two state-of-the-art deep learning models: a text-promptable, open-set object detector and a high-performance, promptable segmentation model.

### **GroundingDINO: Open-Set Object Detection**
GroundingDINO serves as the **region proposal network** in this pipeline. Unlike traditional object detectors that are limited to a fixed set of pre-trained classes, GroundingDINO is an **open-set detector**. It can localize objects in an image based on arbitrary, free-form text prompts. Its architecture consists of an image backbone, a text backbone, and a feature enhancer module that allows for deep fusion of visual and linguistic features. This enables it to understand the relationship between a textual description (e.g., "a person on a chair") and the corresponding spatial regions within the image, which it outputs as bounding boxes.

### **Segment Anything Model 2 (SAM 2): Promptable High-Fidelity Segmentation**
SAM 2 is a foundational model for segmentation, capable of generating exceptionally precise masks for a wide variety of objects and image types. Its key innovation is its promptable design; it can accept various forms of input—such as points, bounding boxes, or rough masks—to specify which object to segment. For this project, its most critical capability is its advanced architecture for **video object segmentation**. The specialized `SAM2VideoPredictor` is engineered to maintain temporal consistency, allowing it to take an initial mask or box on one frame and efficiently propagate it to subsequent frames, tracking the object with high fidelity even through motion and minor appearance changes.

---

## ⚙️ Workflow & Pipeline Architecture
The project is architecturally divided into two distinct, sequential workflows: one for static image segmentation and another for the more complex video segmentation task.

### **Part 1: Text-Prompted Image Segmentation Pipeline**

1.  **Environment Configuration & Dependency Installation:** The workflow begins by preparing the computational environment. This involves cloning the official repositories for both `Grounded-Segment-Anything` and `segment-anything-2`, followed by a comprehensive installation of all required Python packages and dependencies as specified by their respective requirements files.

2.  **Acquisition of Pre-trained Model Assets:** The pipeline downloads the necessary pre-trained model weights. This includes the `groundingdino_swint_ogc.pth` checkpoint for the object detector and the `sam2.1_hiera_large.pt` checkpoint for the segmentation model, which are stored locally for inference.

3.  **Model Initialization and Loading:** The GroundingDINO and SAM 2 models are instantiated and loaded onto the target computational device, preferably a CUDA-enabled GPU to accelerate inference. A crucial technical step involves temporarily changing the working directory to the SAM 2 repository to allow its internal configuration management system (Hydra) to correctly parse model configuration files. The `SAM2ImagePredictor` is specifically loaded for this static image task.

4.  **Prompt-Driven Object Detection (Seeding):** With the models loaded, the user provides a source image and a natural language text prompt. The image and prompt are fed into the GroundingDINO model. It performs inference to identify and localize all objects matching the text description, producing a set of bounding boxes, confidence scores, and the corresponding text phrases. The `BOX_THRESHOLD` and `TEXT_THRESHOLD` hyperparameters are used here to filter out low-confidence detections.

5.  **Coordinate System Transformation:** A critical intermediate step is the transformation of bounding box coordinates. GroundingDINO outputs boxes in a normalized, center-based format (`[center_x, center_y, width, height]`). SAM 2, however, requires absolute, pixel-based coordinates in a corner-based format (`[x1, y1, x2, y2]`). This step precisely denormalizes and converts the coordinate system to ensure compatibility between the two models.

6.  **Promptable High-Fidelity Segmentation:** The source image is first set within the `SAM2ImagePredictor`'s state. Then, the transformed bounding boxes from the previous step are passed as prompts to the predictor's `predict` method. SAM 2 uses these boxes as spatial cues to generate highly detailed and accurate segmentation masks for each detected object.

7.  **Results Synthesis & Visualization:** To produce the final output, the `supervision` library is employed. A `MaskAnnotator` is used to overlay the generated masks onto the original image, with distinct colors for each instance. A `LabelAnnotator` then draws the corresponding text labels and confidence scores. The final annotated image, presenting a clear visual summary of the text-prompted segmentation, is then displayed.

### **Part 2: Bonus - Text-Driven Video Object Segmentation Pipeline**

1.  **Temporal Data Preparation (Frame Extraction):** The video workflow begins by pre-processing the source video file. The entire video is decomposed into a sequential series of individual image frames, which are saved to a dedicated directory. This atomic representation allows the model to process the video frame-by-frame.

2.  **Specialized Video Model Initialization:** A key architectural change is made here. Instead of the image predictor, the `build_sam2_video_predictor` is instantiated. This model variant is specifically designed for temporal tasks and contains mechanisms for maintaining object identity and propagating masks across frames.

3.  **Initial Frame Seeding and Object Grounding:** The core strategy of this pipeline is to avoid redundant detection on every frame. GroundingDINO is run **only once** on a single, user-specified frame from the video (e.g., the first frame or a frame where the object is clearly visible). This single detection pass generates the initial bounding box(es) for the target object(s) described in the text prompt.

4.  **Video State Initialization and Prompting:** The SAM 2 video predictor's inference state is initialized with the directory of extracted frames. The bounding box(es) from the initial seeding step are then programmatically added to the predictor's state for that specific frame index. During this process, each object is assigned a unique `object_id` which will be used for tracking.

5.  **Automated Mask Propagation and Tracking:** This is the central computational step of the video pipeline. The `propagate_in_video` function is invoked. This powerful method iterates through the entire sequence of frames, starting from the initial seeded frame. Using the information from the first prompt, it autonomously **tracks and segments** the designated objects in all subsequent frames without requiring any further user input. It intelligently handles object motion and appearance changes to maintain a consistent mask over time.

6.  **Annotated Video Rendering:** As the model propagates the masks, each frame is processed in real-time. The tracked masks and their corresponding `tracker_id` labels are annotated onto each frame. These annotated frames are then passed to a `VideoSink`, which re-encodes the sequence of images back into a final, cohesive output video file. The resulting video shows the target objects being continuously segmented and tracked throughout the clip.

---

A video in the repo is present which is the output when the prompt is "car" for a video sequence. As you can see the car is segmented in the vieeo.
