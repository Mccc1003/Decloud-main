# Physical-based Diffusion Model for Remote Sensing Cloud and Hazy Removal 

C. Ma and G. Liu, "Physics-based Diffusion Model for Joint Cloud and Haze Removal in Remote Sensing Images," 2025 8th International Conference on Computer Information Science and Application Technology (CISAT), Kunming, China, 2025, pp. 586-589, doi: 10.1109/CISAT66811.2025.11181898.


Abstract:
Remote sensing images are often affected by cloud and haze, which degrade surface observation accuracy. Existing deep learning methods focus on removing cloud or haze independently, lacking a unified approach for simultaneous processing. To address this limitation, we propose a physics-based diffusion model for joint cloud and haze removal from remote sensing images. Given the interrelated nature of cloud and haze and the challenges in decoupling them with existing physical models, our approach combines the atmospheric scattering model and cloud degradation model to construct a unified framework. A Transformer network learns the physical parameters for initial removal, and these parameters are integrated into a diffusion model, enhanced with a structure extraction module to refine spatial details. Experiments on remote sensing datasets show the effectiveness of our method.


keywords: {Degradation;Visualization;Clouds;Atmospheric modeling;Scattering;Interference;Diffusion models;Transformers;Remote sensing;Surface treatment;Remote sensing;physical parameters;diffusion model;cloud removal},


<img width="755" height="436" alt="image" src="https://github.com/user-attachments/assets/51675460-8162-40a0-969e-a1010fea2c6e" />
Fig. 1. Overall Architecture of the Framework. 


<img width="1088" height="812" alt="图片2-1" src="https://github.com/user-attachments/assets/96129fcb-a8fa-4466-9e86-52194ae2bb42" />
Fig. 2. Estimation of physical parameters for cloud image.


<img width="1257" height="903" alt="图片3-1" src="https://github.com/user-attachments/assets/092970c7-1357-4678-baa2-e8ffcc43ac07" />
Fig. 3. Illustration of image structural map.


<img width="902" height="642" alt="图片4" src="https://github.com/user-attachments/assets/7812c443-f8ce-40c2-b67e-a36a88c39756" />
Fig. 4. Comparison of visual results on SateHaze1k dataset. 


<img width="898" height="405" alt="图片5" src="https://github.com/user-attachments/assets/92f8cf3c-0997-4a74-b219-b6e7b31d589b" />
Fig. 5. Comparison of visual results on T-Cloud dataset.


<img width="898" height="404" alt="图片6" src="https://github.com/user-attachments/assets/d4255a56-c0dd-4e85-9d0c-0f836a6e53e5" />
Fig. 6. Comparison of visual results on RICE2 dataset. 


<img width="768" height="315" alt="image" src="https://github.com/user-attachments/assets/83351e14-3cbf-4e14-8054-248180d1624e" />


<img width="764" height="319" alt="image" src="https://github.com/user-attachments/assets/2dc0417d-ccf9-41ff-81aa-1b7bdfb74baf" />
