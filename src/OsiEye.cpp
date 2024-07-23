/*******************************************************
 * Open Source for Iris : OSIRIS
 * Version : 4.0
 * Date : 2011
 * Author : Guillaume Sutra, Telecom SudParis, France
 * License : BSD
 ********************************************************/

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <fstream>
#include <stdexcept>
#include "cv.h"
#include "OsiEye.h"
#include "OsiProcessings.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include "../include/base64.hpp"

using namespace std;

namespace osiris
{

    // CONSTRUCTORS & DESTRUCTORS
    /////////////////////////////

    OsiEye::OsiEye()
    {
        mpOriginalImage = 0;
        mpOriginalImageBase64 = 0;
        mpSegmentedImage = 0;
        mpMask = 0;
        mpNormalizedImage = 0;
        mpNormalizedMask = 0;
        mpIrisCode = 0;
        mPupil.setCircle(0, 0, 0);
        mIris.setCircle(0, 0, 0);
    }

    OsiEye::~OsiEye()
    {
        cvReleaseImage(&mpOriginalImage);
        cvReleaseImage(&mpOriginalImageBase64);
        cvReleaseImage(&mpSegmentedImage);
        cvReleaseImage(&mpMask);
        cvReleaseImage(&mpNormalizedImage);
        cvReleaseImage(&mpNormalizedMask);
        cvReleaseImage(&mpIrisCode);
    }

    // Functions for loading images and parameters
    //////////////////////////////////////////////

    void OsiEye::loadImage(const string &rFilename, IplImage **ppImage)
    {
        // :WARNING: ppImage is a pointer of pointer
        try
        {
            if (*ppImage)
            {
                cvReleaseImage(ppImage);
            }

            *ppImage = cvLoadImage(rFilename.c_str(), 0);
            if (!*ppImage)
            {
                cout << "Cannot load image : " << rFilename << endl;
            }
        }
        catch (exception &e)
        {
            cout << e.what() << endl;
        }
    }

    void OsiEye::loadImageFromBase64(const string &rFilename, IplImage **ppImage)
    {
        try
        {
            if (*ppImage)
            {
                cvReleaseImage(ppImage);
            }

            // Read the file content
            ifstream file(rFilename);
            if (!file)
            {
                cout << "Cannot open file : " << rFilename << endl;
                return;
            }

            stringstream buffer;
            buffer << file.rdbuf();
            string base64String = buffer.str();
            file.close();

            // Decode base64 string to bytes
            auto decodedData = base64::from_base64(base64String);
            std::cout << "Decoded data size: " << decodedData.size() << std::endl;

            // Convert decoded data to cv::Mat
            vector<uchar> data(decodedData.begin(), decodedData.end());
            cv::Mat img = cv::imdecode(data, cv::IMREAD_GRAYSCALE);

            if (img.empty())
            {
                cout << "Cannot decode base64 image data" << endl;
                return;
            }
            else
            {
                cout << "Image decoded from base64 data" << endl;
            }

            // Convert cv::Mat to IplImage
            *ppImage = cvCreateImage(cvSize(img.cols, img.rows), IPL_DEPTH_8U, img.channels());
            if (*ppImage)
            {
                memcpy((*ppImage)->imageData, img.data, img.total() * img.elemSize());
            }
            else
            {
                cout << "Cannot create IplImage from base64 data" << endl;
            }

            if (!*ppImage)
            {
                cout << "Cannot load image from base64 data" << endl;
            }
            else
            {
                cout << "Image loaded from base64 data" << endl;
            }
        }
        catch (exception &e)
        {
            cout << e.what() << endl;
        }
    }

    void OsiEye::loadOriginalImage(const string &rFilename)
    {
        loadImage(rFilename, &mpOriginalImage);
    }

    void OsiEye::loadOriginalImageFromBase64(const string &rFilename)
    {
        // loadImageFromBase64(rFilename, &mpOriginalImageBase64);
        loadImageFromBase64(rFilename, &mpOriginalImage);
    }

    void OsiEye::loadMask(const string &rFilename)
    {
        loadImage(rFilename, &mpMask);
    }

    void OsiEye::loadNormalizedImage(const string &rFilename)
    {
        loadImage(rFilename, &mpNormalizedImage);
    }

    void OsiEye::loadNormalizedMask(const string &rFilename)
    {
        loadImage(rFilename, &mpNormalizedMask);
    }

    void OsiEye::loadIrisCode(const string &rFilename)
    {
        loadImage(rFilename, &mpIrisCode);
    }

    void OsiEye::loadParameters(const string &rFilename)
    {
        // Open the file
        ifstream file(rFilename.c_str(), ios::in);

        // If file is not opened
        if (!file)
        {
            throw runtime_error("Cannot load the parameters in " + rFilename);
        }
        try
        {
            // int xp , yp , rp , xi , yi , ri ;
            // file >> xp ;
            // file >> yp ;
            // file >> rp ;
            // file >> xi ;
            // file >> yi ;
            // file >> ri ;
            // mPupil.setCircle(xp,yp,rp) ;
            // mIris.setCircle(xi,yi,ri) ;
            int nbp = 0;
            int nbi = 0;
            file >> nbp;
            file >> nbi;
            mThetaCoarsePupil.resize(nbp, 0.0);
            mThetaCoarseIris.resize(nbi, 0.0);
            mCoarsePupilContour.resize(nbp, cvPoint(0, 0));
            mCoarseIrisContour.resize(nbi, cvPoint(0, 0));
            // matrix.resize( num_of col , vector<double>( num_of_row , init_value ) );
            for (int i = 0; i < nbp; i++)
            {
                file >> mCoarsePupilContour[i].x;
                file >> mCoarsePupilContour[i].y;
                file >> mThetaCoarsePupil[i];
            }
            for (int j = 0; j < nbi; j++)
            {
                file >> mCoarseIrisContour[j].x;
                file >> mCoarseIrisContour[j].y;
                file >> mThetaCoarseIris[j];
            }
        }
        catch (exception &e)
        {
            cout << e.what() << endl;
            throw runtime_error("Error while loading parameters from " + rFilename);
        }

        // Close the file
        file.close();
    }

    // Functions for saving images and parameters
    /////////////////////////////////////////////

    void OsiEye::saveImage(const string &rFilename, const IplImage *pImage)
    {
        // :TODO: no exception here, but 2 error messages
        // 1. pImage does NOT exist => "image was neither comptued nor loaded"
        // 2. cvSaveImage returns <=0 => "rFilename = invalid for saving"
        if (!pImage)
        {
            throw runtime_error("Cannot save image " + rFilename + " because this image is not built");
        }
        if (!cvSaveImage(rFilename.c_str(), pImage))
        {
            cout << "Cannot save image as " << rFilename << endl;
        }
    }

    void OsiEye::saveSegmentedImage(const string &rFilename)
    {
        saveImage(rFilename, mpSegmentedImage);
    }

    void OsiEye::saveMask(const string &rFilename)
    {
        saveImage(rFilename, mpMask);
    }

    void OsiEye::saveNormalizedImage(const string &rFilename)
    {
        saveImage(rFilename, mpNormalizedImage);
    }

    void OsiEye::saveNormalizedMask(const string &rFilename)
    {
        saveImage(rFilename, mpNormalizedMask);
    }

    void OsiEye::saveIrisCode(const string &rFilename)
    {
        saveImage(rFilename, mpIrisCode);
    }

    void OsiEye::saveParameters(const string &rFilename)
    {
        // Open the file
        ofstream file(rFilename.c_str(), ios::out);

        // If file is not opened
        if (!file)
        {
            throw runtime_error("Cannot save the parameters in " + rFilename);
        }

        try
        {
            //    file << mPupil.getCenter().x << " " ;
            //    file << mPupil.getCenter().y << " " ;
            //    file << mPupil.getRadius() << endl ;
            //    file << mIris.getCenter().x << " " ;
            //    file << mIris.getCenter().y << " " ;
            //    file << mIris.getRadius() << endl ;
            file << mCoarsePupilContour.size() << endl;
            file << mCoarseIrisContour.size() << endl;
            for (int i = 0; i < (mCoarsePupilContour.size()); i++)
            {
                file << mCoarsePupilContour[i].x << " ";
                file << mCoarsePupilContour[i].y << " ";
                file << mThetaCoarsePupil[i] << " ";
            }
            file << endl;
            for (int j = 0; j < (mCoarseIrisContour.size()); j++)
            {
                file << mCoarseIrisContour[j].x << " ";
                file << mCoarseIrisContour[j].y << " ";
                file << mThetaCoarseIris[j] << " ";
            }
        }
        catch (exception &e)
        {
            cout << e.what() << endl;
            throw runtime_error("Error while saving parameters in " + rFilename);
        }

        // Close the file
        file.close();
    }

    // Functions for processings
    ////////////////////////////

    void OsiEye::initMask()
    {
        if (mpMask)
        {
            cvReleaseImage(&mpMask);
        }
        if (!mpOriginalImage)
        {
            throw runtime_error("Cannot initialize the mask because original image is not loaded");
        }
        mpMask = cvCreateImage(cvGetSize(mpOriginalImage), IPL_DEPTH_8U, 1);
        cvSet(mpMask, cvScalar(255));
    }

    void OsiEye::segment(int minIrisDiameter, int minPupilDiameter, int maxIrisDiameter, int maxPupilDiameter)
    {
        if (!mpOriginalImage)
        {
            throw std::runtime_error("Cannot segment image because original image is not loaded");
        }

        // Print input parameters
        std::cout << "minIrisDiameter: " << minIrisDiameter << std::endl;
        std::cout << "minPupilDiameter: " << minPupilDiameter << std::endl;
        std::cout << "maxIrisDiameter: " << maxIrisDiameter << std::endl;
        std::cout << "maxPupilDiameter: " << maxPupilDiameter << std::endl;

        // Verify the original image dimensions
        std::cout << "Original Image Size: " << mpOriginalImage->width << "x" << mpOriginalImage->height << std::endl;

        // Initialize mask and segmented image
        mpMask = cvCreateImage(cvGetSize(mpOriginalImage), IPL_DEPTH_8U, 1);
        if (!mpMask)
        {
            std::cerr << "Failed to create mask image" << std::endl;
        }
        else
        {
            std::cout << "Mask image created successfully" << std::endl;
        }

        mpSegmentedImage = cvCreateImage(cvGetSize(mpOriginalImage), IPL_DEPTH_8U, 3);
        if (!mpSegmentedImage)
        {
            std::cerr << "Failed to create segmented image" << std::endl;
        }
        else
        {
            std::cout << "Segmented image created successfully" << std::endl;
        }

        cvCvtColor(mpOriginalImage, mpSegmentedImage, CV_GRAY2BGR);

        // Print intermediate values
        std::cout << "mpOriginalImage size: " << cvGetSize(mpOriginalImage).width << "x" << cvGetSize(mpOriginalImage).height << std::endl;
        std::cout << "mpMask size: " << cvGetSize(mpMask).width << "x" << cvGetSize(mpMask).height << std::endl;
        std::cout << "mpSegmentedImage size: " << cvGetSize(mpSegmentedImage).width << "x" << cvGetSize(mpSegmentedImage).height << std::endl;

        // Processing functions
        OsiProcessings op;

        // Segment the eye
        op.segment(mpOriginalImage, mpMask, mPupil, mIris, mThetaCoarsePupil, mThetaCoarseIris, mCoarsePupilContour, mCoarseIrisContour, minIrisDiameter, minPupilDiameter, maxIrisDiameter, maxPupilDiameter);

        // Print values after segmentation
        std::cout << "mPupil center: (" << mPupil.getCenter().x << ", " << mPupil.getCenter().y << "), radius: " << mPupil.getRadius() << std::endl;
        std::cout << "mIris center: (" << mIris.getCenter().x << ", " << mIris.getCenter().y << "), radius: " << mIris.getRadius() << std::endl;

        // Draw on segmented image
        IplImage *tmp = cvCloneImage(mpMask);
        cvZero(tmp);
        cvCircle(tmp, mIris.getCenter(), mIris.getRadius(), cvScalar(255), -1);
        cvCircle(tmp, mPupil.getCenter(), mPupil.getRadius(), cvScalar(0), -1);
        cvSub(tmp, mpMask, tmp);
        cvSet(mpSegmentedImage, cvScalar(0, 0, 255), tmp);
        cvReleaseImage(&tmp);
        cvCircle(mpSegmentedImage, mPupil.getCenter(), mPupil.getRadius(), cvScalar(0, 255, 0));
        cvCircle(mpSegmentedImage, mIris.getCenter(), mIris.getRadius(), cvScalar(0, 255, 0));

        // Print final segmented image info
        std::cout << "Segmentation complete" << std::endl;
    }

    void OsiEye::normalize(int rWidthOfNormalizedIris, int rHeightOfNormalizedIris)
    {
        // Processing functions
        OsiProcessings op;

        // For the image
        if (!mpOriginalImage)
        {
            throw runtime_error("Cannot normalize image because original image is not loaded");
        }

        mpNormalizedImage = cvCreateImage(cvSize(rWidthOfNormalizedIris, rHeightOfNormalizedIris), IPL_DEPTH_8U, 1);

        if (mThetaCoarsePupil.empty() || mThetaCoarseIris.empty())
        {
            // throw runtime_error("Cannot normalize image because circles are not correctly computed") ;
            throw runtime_error("Cannot normalize image because contours are not correctly computed/loaded");
        }

        // op.normalize(mpOriginalImage,mpNormalizedImage,mPupil,mIris) ;
        op.normalizeFromContour(mpOriginalImage, mpNormalizedImage, mPupil, mIris, mThetaCoarsePupil, mThetaCoarseIris, mCoarsePupilContour, mCoarseIrisContour);

        // For the mask
        if (!mpMask)
        {
            initMask();
        }

        mpNormalizedMask = cvCreateImage(cvSize(rWidthOfNormalizedIris, rHeightOfNormalizedIris), IPL_DEPTH_8U, 1);

        // op.normalize(mpMask,mpNormalizedMask,mPupil,mIris) ;
        op.normalizeFromContour(mpMask, mpNormalizedMask, mPupil, mIris, mThetaCoarsePupil, mThetaCoarseIris, mCoarsePupilContour, mCoarseIrisContour);
    }

    void OsiEye::encode(const vector<CvMat *> &rGaborFilters)
    {
        if (!mpNormalizedImage)
        {
            throw runtime_error("Cannot encode because normalized image is not loaded");
        }

        // Create the image to store the iris code
        CvSize size = cvGetSize(mpNormalizedImage);
        mpIrisCode = cvCreateImage(cvSize(size.width, size.height * rGaborFilters.size()), IPL_DEPTH_8U, 1);

        // Encode
        OsiProcessings op;
        op.encode(mpNormalizedImage, mpIrisCode, rGaborFilters);
    }

    float OsiEye::match(OsiEye &rEye, const CvMat *pApplicationPoints)
    {
        // Check that both iris codes are built
        if (!mpIrisCode)
        {
            throw runtime_error("Cannot match because iris code 1 is not built (nor computed neither loaded)");
        }
        if (!rEye.mpIrisCode)
        {
            throw runtime_error("Cannot match because iris code 2 is not built (nor computed neither loaded)");
        }

        // Initialize the normalized masks
        // :TODO: must inform the user of this step, for example if user provides masks for all images
        // but one is missing for only one image. However, message must not be spammed if the user
        // did not provide any mask ! So it must be found a way to inform user but without spamming
        if (!mpNormalizedMask)
        // if (true)
        {
            mpNormalizedMask = cvCreateImage(cvGetSize(pApplicationPoints), IPL_DEPTH_8U, 1);
            cvSet(mpNormalizedMask, cvScalar(255));
            cout << "Normalized mask of image 1 is missing for matching. All pixels are initialized to 255" << endl;
        }
        else
        {
            cout << "Normalized mask of image 1 is available for matching" << endl;
        }
        if (!rEye.mpNormalizedMask)
        // if (true)
        {
            rEye.mpNormalizedMask = cvCreateImage(cvGetSize(pApplicationPoints), IPL_DEPTH_8U, 1);
            cvSet(rEye.mpNormalizedMask, cvScalar(255));
            cout << "Normalized mask of image 2 is missing for matching. All pixels are initialized to 255" << endl;
        }
        else
        {
            cout << "Normalized mask of image 2 is available for matching" << endl;
        }

        // Build the total mask = mask1 * mask2 * points
        IplImage *temp = cvCreateImage(cvGetSize(pApplicationPoints), mpIrisCode->depth, 1);
        cvSet(temp, cvScalar(0));
        cvAnd(mpNormalizedMask, rEye.mpNormalizedMask, temp, pApplicationPoints);

        // Copy the mask f times, where f correspond to the number of codes (= number of filters)
        int n_codes = mpIrisCode->height / pApplicationPoints->height;
        IplImage *total_mask = cvCreateImage(cvGetSize(mpIrisCode), IPL_DEPTH_8U, 1);
        for (int n = 0; n < n_codes; n++)
        {
            cvSetImageROI(total_mask, cvRect(0, n * pApplicationPoints->height, pApplicationPoints->width, pApplicationPoints->height));
            cvCopy(temp, total_mask);
            cvResetImageROI(total_mask);
        }

        // Match
        OsiProcessings op;
        int numUnmaskedBits;
        float rawScore = op.match(mpIrisCode, rEye.mpIrisCode, total_mask, numUnmaskedBits);
        std::cout << "Raw score: " << rawScore << std::endl;
        std::cout << "Number of unmasked bits: " << numUnmaskedBits << std::endl;

        // Normalize the Hamming distance
        // TODO: Search for the average number of unmasked bits in the database
        float normalizedScore = 0.5 - (0.5 - rawScore) * std::sqrt(static_cast<float>(numUnmaskedBits) / 230000.0f);
        std::cout << "Normalized score: " << normalizedScore << std::endl;
        // Free memory
        cvReleaseImage(&temp);
        cvReleaseImage(&total_mask);

        return rawScore;
        // return normalizedScore;
    }

    // Function to read buffers and metadata from a file
    std::pair<IplImage *, IplImage *> OsiEye::read_buffers_from_file(const std::string &input_file)
    {
        std::ifstream input(input_file, std::ios::binary);
        if (!input)
        {
            throw std::runtime_error("Failed to open file for reading: " + input_file);
        }

        // Read width, height, and channels for the first image
        int width1, height1, channels1;
        input.read(reinterpret_cast<char *>(&width1), sizeof(width1));
        input.read(reinterpret_cast<char *>(&height1), sizeof(height1));
        input.read(reinterpret_cast<char *>(&channels1), sizeof(channels1));

        std::cout << "Width 1: " << width1 << std::endl;
        std::cout << "Height 1: " << height1 << std::endl;

        // Read the first image buffer
        size_t buffer1_size = width1 * height1 * channels1;
        std::vector<uint8_t> buffer1(buffer1_size);
        input.read(reinterpret_cast<char *>(buffer1.data()), buffer1.size());

        std::cout << "Buffer 1 size: " << buffer1.size() << std::endl;

        // Read delimiter
        const char delimiter[] = "\0\0\0\0DELIMITER\0\0\0\0";
        size_t delimiter_size = sizeof(delimiter);
        char delimiter_read[delimiter_size];
        input.read(delimiter_read, delimiter_size);
        if (std::memcmp(delimiter_read, delimiter, delimiter_size) != 0)
        {
            throw std::runtime_error("Delimiter not found or incorrect in file.");
        }

        // Read width, height, and channels for the second image
        int width2, height2, channels2;
        input.read(reinterpret_cast<char *>(&width2), sizeof(width2));
        input.read(reinterpret_cast<char *>(&height2), sizeof(height2));
        input.read(reinterpret_cast<char *>(&channels2), sizeof(channels2));

        std::cout << "Width 2: " << width2 << std::endl;
        std::cout << "Height 2: " << height2 << std::endl;


        // Read the second image buffer
        size_t buffer2_size = width2 * height2 * channels2;
        std::vector<uint8_t> buffer2(buffer2_size);
        input.read(reinterpret_cast<char *>(buffer2.data()), buffer2.size());

        // Create IplImage for the first image
        IplImage *image1 = cvCreateImage(cvSize(width1, height1), IPL_DEPTH_8U, channels1);
        std::memcpy(image1->imageData, buffer1.data(), buffer1.size());

        // Create IplImage for the second image
        IplImage *image2 = cvCreateImage(cvSize(width2, height2), IPL_DEPTH_8U, channels2);
        std::memcpy(image2->imageData, buffer2.data(), buffer2.size());

        return std::make_pair(image1, image2);
    }
    
    float OsiEye::matchFromBuffer(const std::string &filename1, const std::string &filename2, const CvMat *pApplicationPoints)
    {
        // print size of pApplicationPoints
        std::cout << "Application points size: " << pApplicationPoints->width << "x" << pApplicationPoints->height << std::endl;
        std::cout << "Matching from buffer" << std::endl;
        // Read iris codes and masks from files
        auto [irisCode1, normalizedMask1] = read_buffers_from_file(filename1);
        auto [irisCode2, normalizedMask2] = read_buffers_from_file(filename2);

        std::cout << "Iris code 1 size: " << irisCode1->width << std::endl;
        std::cout << "Iris code 2 size: " << irisCode2->width << std::endl;

        if (!irisCode1 || !irisCode2)
        {
            throw std::runtime_error("Cannot match because one or both iris codes could not be read from file");
        }

        // Use default mask if not provided
        if (!normalizedMask1)
        {
            normalizedMask1 = cvCreateImage(cvGetSize(pApplicationPoints), IPL_DEPTH_8U, 1);
            cvSet(normalizedMask1, cvScalar(255));
            std::cout << "Normalized mask of image 1 is missing. All pixels are initialized to 255" << std::endl;
        }
        if (!normalizedMask2)
        {
            normalizedMask2 = cvCreateImage(cvGetSize(pApplicationPoints), IPL_DEPTH_8U, 1);
            cvSet(normalizedMask2, cvScalar(255));
            std::cout << "Normalized mask of image 2 is missing. All pixels are initialized to 255" << std::endl;
        }
        if (!normalizedMask1 || normalizedMask1->width != pApplicationPoints->width || normalizedMask1->height != pApplicationPoints->height)
        {
            IplImage *resizedMask1 = cvCreateImage(cvGetSize(pApplicationPoints), IPL_DEPTH_8U, 1);
            cvResize(normalizedMask1, resizedMask1);
            cvReleaseImage(&normalizedMask1);
            normalizedMask1 = resizedMask1;
        }

        if (!normalizedMask2 || normalizedMask2->width != pApplicationPoints->width || normalizedMask2->height != pApplicationPoints->height)
        {
            IplImage *resizedMask2 = cvCreateImage(cvGetSize(pApplicationPoints), IPL_DEPTH_8U, 1);
            cvResize(normalizedMask2, resizedMask2);
            cvReleaseImage(&normalizedMask2);
            normalizedMask2 = resizedMask2;
        }

        // start time
        auto start = std::chrono::high_resolution_clock::now();

        // Build the total mask
        IplImage *temp = cvCreateImage(cvGetSize(pApplicationPoints), irisCode1->depth, 1);
        cvSet(temp, cvScalar(0));
        cvAnd(normalizedMask1, normalizedMask2, temp, pApplicationPoints);

        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed1 = end1 - start;
        std::cout << "Elapsed time for building total mask: " << elapsed1.count() << " ms" << std::endl;

        std::cout << "Total mask size: " << temp->width << "x" << temp->height << std::endl;

        // Copy the mask for each code
        int n_codes = irisCode1->height / pApplicationPoints->height;
        IplImage *total_mask = cvCreateImage(cvGetSize(irisCode1), IPL_DEPTH_8U, 1);
        for (int n = 0; n < n_codes; n++)
        {
            cvSetImageROI(total_mask, cvRect(0, n * pApplicationPoints->height, pApplicationPoints->width, pApplicationPoints->height));
            cvCopy(temp, total_mask);
            cvResetImageROI(total_mask);
        }
        std::cout << "Size of total mask: " << total_mask->width << "x" << total_mask->height << std::endl;

        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed2 = end2 - end1;
        std::cout << "Elapsed time for copying total mask: " << elapsed2.count() << " ms" << std::endl;

        // Match
        OsiProcessings op;
        int numUnmaskedBits;
        float rawScore = op.match(irisCode1, irisCode2, total_mask, numUnmaskedBits);
        std::cout << "Raw score: " << rawScore << std::endl;
        std::cout << "Number of unmasked bits: " << numUnmaskedBits << std::endl;

        // Normalize the Hamming distance
        float normalizedScore = 0.5 - (0.5 - rawScore) * std::sqrt(static_cast<float>(numUnmaskedBits) / 230000.0f);
        std::cout << "Normalized score: " << normalizedScore << std::endl;

        // end time
        auto end = std::chrono::high_resolution_clock::now();
        // elapsed as ms
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl;

        // Free memory
        cvReleaseImage(&temp);
        cvReleaseImage(&total_mask);
        cvReleaseImage(&irisCode1);
        cvReleaseImage(&irisCode2);
        cvReleaseImage(&normalizedMask1);
        cvReleaseImage(&normalizedMask2);

        return rawScore;
        // return normalizedScore;
    }

} // end of namespace