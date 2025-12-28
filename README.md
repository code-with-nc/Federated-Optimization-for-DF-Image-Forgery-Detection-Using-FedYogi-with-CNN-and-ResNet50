# Federated-Optimization-for-DF-Image-Forgery-Detection-Using-FedYogi-with-CNN-and-ResNet50
A privacy-preserving image forgery detection framework using Federated Learning. Combines FedYogi optimization with CNN and ResNet50 to handle non-IID forensic data. Trains models collaboratively without sharing raw images, improving stability and accuracy on CASIA 1.0.

# CASIA 1.0 DataSet
*https://drive.google.com/file/d/14f3jU2VsxTYopgSE1Vvv4hMlFvpAKHUY/view


# Authentic images: 800 images (8 categories, 100 images in each category).

Au_ani_0001.jpg

Au: Authentic

ani: animal category

Other categories: arc (architecture), art, cha (characters), nat (nature), pla (plants), sec, txt (texture)

## Tampering images

a. Spliced image

        Sp_D_CND_A_pla0005_pla0023_0281.jpg
* Sp: Splicing
* D: Different (means the tampered region was copied from the different image)
* Next 4 letters stand for the techniques they used to create the images. Unfortunately, I don't remember exactly.
* pla0005: the source image
* pla0023: the target image
* 0281: tampered image ID

b. Copy-move images

        Sp_S_CND_A_pla0016_pla0016_0196.jpg
* Sp: Tampering
* S: Same (means the tampered region was copied from the same image)
* And the rest is similar to case a.

If you use the groundtruth dataset for a scientific publication, please cite the following papers

* CASIA dataset

        @inproceedings{Dong2013,
        doi = {10.1109/chinasip.2013.6625374},
        url = {https://doi.org/10.1109/chinasip.2013.6625374},
        year = {2013},
        month = jul,
        publisher = {{IEEE}},
        author = {Jing Dong and Wei Wang and Tieniu Tan},
        title = {{CASIA} Image Tampering Detection Evaluation Database},
        booktitle = {2013 {IEEE} China Summit and International Conference on Signal and Information Processing}
        }


 * CASIA groundtruth dataset 
 
        @article{pham2019hybrid,
        title={Hybrid Image-Retrieval Method for Image-Splicing Validation},
        author={Pham, Nam Thanh and Lee, Jong-Weon and Kwon, Goo-Rak and Park, Chun-Su},
        journal={Symmetry},
        volume={11},
        number={1},
        pages={83},
        year={2019},
        publisher={Multidisciplinary Digital Publishing Institute}
        }
