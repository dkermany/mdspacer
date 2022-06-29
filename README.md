<!-- ![Alt text](/Bone_Vessel_Segmentation_Diagram.png?raw=true)-->
<div id="top"></div>
<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- PROJECT LOGO -->
<div align="center">
<!--   <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

  <h3 align="center">BoneMetsAI</h3>

  <p align="center">
    A library for the analyses of microscopy images of breast cancer metastases in mouse femur models  
    <br />
    <a href="#">View Demo</a>
    ·
    <a href="https://github.com/dkermany/BoneMetsAI/issues">Report Bug</a>
    ·
    <a href="https://github.com/dkermany/BoneMetsAI/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
## Table of Contents
<ol>
  <li>
    <a href="#project-overview">About The Project</a>
  </li>
  <li>
    <a href="#getting-started">Getting Started</a>
    <ul>
      <li><a href="#prerequisites">Prerequisites</a></li>
    </ul>
  </li>
  <li><a href="#usage">Usage</a></li>
  <li><a href="#roadmap">Roadmap</a></li>
  <li><a href="#acknowledgments">Acknowledgments</a></li>
</ol>

<!-- ABOUT THE PROJECT -->
## Project Overview

### Objectives

![Alt text](/cover_image.png?raw=true)

Determine if there is any morphological changes to bone blood vessels in the vicinity of identified metastasized breast cancer cells, as well as any spatio-temporal relationship between NG2+ endothelial cells and the tumor cells.

In order to do so, the following plan was developed:
TODO
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap
- [x] Reproduce previous work in matlab
- [x] Manually labeled 2D samples
- [ ] Segmentation Datasets Loaded
  - [x] 2D Bone Microscope slices (JPG/PNG)
  - [x] [COCO](https://cocodataset.org/) Data Preprocessing
  - [ ] [CoNSeP](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/) Data Preprocessing
  - [ ] 3D Bone Microscope slices (OIF/OIB/TIFF)
- [ ] Segmentation
  - [ ] Reproduce previous work
  - [ ] Write evaluation functions
    - [ ] Class-weighted pixelwise accuracy
    - [ ] Class-weighted jaccard score
    - [ ] Class-weighted dice score
  - [ ] Write UNet Base Model
    - [ ] Train weights on COCO Dataset
    - [ ] Continue training weights on CoNSeP Dataset
    - [ ] Train on half of bone images
  - [ ] Compare with previous work
  - [ ] Try with ImageNet pretrained encoder-decoders
- [ ] Classification Datasets Loaded
  - [ ] Binary `affected` vs `unaffected` vessels based on tumor cell vicinity
    - [ ] Vessel instance segmentation and bounding box patch generation
  - [ ] Train

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

Libraries and environment setup using the `pipenv` library.

### Prerequisites

Install dependencies
  ```sh
  pipenv install --dev
  ```

<!-- USAGE EXAMPLES -->
## Usage
TODO
- Training
- Evaluation
- Utils
<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTRIBUTING -->
<!-- ## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- LICENSE -->
<!-- ## License

Distributed under the MIT License. See `LICENSE.txt` for more information.
<p align="right">(<a href="#top">back to top</a>)</p> -->


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
* Dr. Stephen Wong
* Dr. Jianting Sheng
* Dr. Weijie Zhang
* Yunjie He
<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
