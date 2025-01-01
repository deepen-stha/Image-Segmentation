#include <iostream>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

class EnhancedFuzzyCMeans {
private:
    int numClusters;
    int maxIterations;
    float epsilon;
    float m;  // fuzziness parameter
    Mat U;    // membership matrix
    Mat segmented;
    bool usePreprocessing;
    
    Mat preprocess(const Mat& input) {
        Mat processed;
        
        // Convert to grayscale first if color image
        Mat gray;
        if (input.channels() > 1) {
            cvtColor(input, gray, COLOR_BGR2GRAY);
        } else {
            input.copyTo(gray);
        }
        
        // Enhance contrast (equalizeHist requires 8UC1)
        Mat enhanced;
        equalizeHist(gray, enhanced);
        
        // Convert to floating point for further processing
        enhanced.convertTo(processed, CV_32F, 1.0/255.0);
        
        // Apply bilateral filter to reduce noise while preserving edges
        Mat bilateral;
        bilateralFilter(processed, bilateral, 9, 75, 75);
        
        // Normalize
        Mat normalized;
        normalize(bilateral, normalized, 0, 1, NORM_MINMAX);
        
        return normalized;
    }
    
    void postprocess(Mat& input) {
        // Create kernel for morphological operations
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
        
        // Apply morphological operations to clean up segmentation
        morphologyEx(input, input, MORPH_OPEN, kernel);
        morphologyEx(input, input, MORPH_CLOSE, kernel);
        
        // Optional: Apply median blur to remove small artifacts
        medianBlur(input, input, 3);
    }

public:
    EnhancedFuzzyCMeans(int clusters = 4, int maxIter = 100, float eps = 1e-4, float fuzziness = 1.8, bool preprocess = true)
        : numClusters(clusters), maxIterations(maxIter), epsilon(eps), m(fuzziness), usePreprocessing(preprocess) {
        
        // Validate parameters
        if (clusters < 2) throw invalid_argument("Number of clusters must be at least 2");
        if (fuzziness <= 1.0) throw invalid_argument("Fuzziness parameter must be greater than 1.0");
        if (maxIter < 1) throw invalid_argument("Maximum iterations must be positive");
    }

    void fit(const Mat& img) {
        // Preprocess the image
        Mat processedImg = usePreprocessing ? preprocess(img) : img;
        
        // Reshape image to 2D array of pixels
        Mat data = processedImg.reshape(1, processedImg.rows * processedImg.cols);
        data.convertTo(data, CV_32F);

        // Initialize membership matrix randomly
        U = Mat::zeros(numClusters, data.rows, CV_32F);
        randu(U, 0, 1);
        normalize(U, U, 1, 0, NORM_L1, -1, Mat());

        // Main FCM loop
        for (int iter = 0; iter < maxIterations; iter++) {
            // Compute cluster centers
            vector<float> centers(numClusters, 0);
            vector<float> center_denoms(numClusters, 0);
            
            for (int k = 0; k < numClusters; k++) {
                for (int i = 0; i < data.rows; i++) {
                    float membership = pow(U.at<float>(k, i), m);
                    centers[k] += membership * data.at<float>(i, 0);
                    center_denoms[k] += membership;
                }
                
                if (center_denoms[k] > numeric_limits<float>::epsilon()) {
                    centers[k] /= center_denoms[k];
                }
            }

            // Update membership matrix
            Mat U_old = U.clone();
            
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < data.rows; i++) {
                for (int k = 0; k < numClusters; k++) {
                    float sum = 0;
                    float d_ki = abs(data.at<float>(i, 0) - centers[k]);
                    
                    if (d_ki < numeric_limits<float>::epsilon()) {
                        U.col(i).setTo(0);
                        U.at<float>(k, i) = 1;
                        continue;
                    }

                    for (int j = 0; j < numClusters; j++) {
                        float d_ji = abs(data.at<float>(i, 0) - centers[j]);
                        if (d_ji > numeric_limits<float>::epsilon()) {
                            sum += pow(d_ki/d_ji, 2/(m-1));
                        }
                    }
                    
                    U.at<float>(k, i) = 1.0f / sum;
                }
            }

            // Check convergence
            double diff = norm(U - U_old);
            if (diff < epsilon) {
                cout << "Converged after " << iter + 1 << " iterations" << endl;
                break;
            }
        }

        // Create segmented image
        segmented = Mat::zeros(data.rows, 1, CV_32F);
        for (int i = 0; i < data.rows; i++) {
            Point maxLoc;
            minMaxLoc(U.col(i), nullptr, nullptr, nullptr, &maxLoc);
            segmented.at<float>(i) = maxLoc.y;
        }
        
        // Reshape back to original image dimensions
        segmented = segmented.reshape(1, img.rows);
        
        // Convert to display format
        segmented.convertTo(segmented, CV_8U, 255.0 / (numClusters - 1));
        
        // Apply post-processing
        postprocess(segmented);
    }

    Mat getSegmented() const {
        return segmented;
    }
    
    // Add method to visualize segmentation with colors
    Mat getColorSegmented() const {
        Mat colorSegmented;
        applyColorMap(segmented, colorSegmented, COLORMAP_JET);
        return colorSegmented;
    }
};

int main() {
    string inputDir = "data/";
    string outputDir = "segmented_output/";

    // Create output directory if it doesn't exist
    if (!fs::exists(outputDir)) {
        fs::create_directory(outputDir);
    }

    auto startTime = chrono::high_resolution_clock::now();

    try {
        for (const auto& entry : fs::directory_iterator(inputDir)) {
            if (entry.path().extension() == ".jpg" || 
                entry.path().extension() == ".png" || 
                entry.path().extension() == ".jpeg") {
                
                string imagePath = entry.path().string();
                cout << "Processing: " << imagePath << endl;

                // Read image
                Mat img = imread(imagePath, IMREAD_COLOR);
                if (img.empty()) {
                    cerr << "Error: Could not read image: " << imagePath << endl;
                    continue;
                }

                // Create FCM instance with optimized parameters
                EnhancedFuzzyCMeans fcm(3, 100, 1e-4, 1.8, true);
                
                // Apply segmentation
                fcm.fit(img);
                
                // Get and save results
                Mat segmented = fcm.getSegmented();
                Mat colorSegmented = fcm.getColorSegmented();
                
                string outputPath = outputDir + entry.path().stem().string() + "_segmented.png";
                string colorOutputPath = outputDir + entry.path().stem().string() + "_segmented_color.png";
                
                imwrite(outputPath, segmented);
                imwrite(colorOutputPath, colorSegmented);
                
                cout << "Saved: " << outputPath << endl;
                cout << "Saved colored version: " << colorOutputPath << endl;
                
                // Display results (uncomment if you want to see results immediately)
                // imshow("Original", img);
                // imshow("Segmented", segmented);
                // imshow("Color Segmented", colorSegmented);
                // waitKey(0);
            }
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return -1;
    }

    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(endTime - startTime).count();
    cout << "Total processing time: " << duration << " seconds" << endl;

    return 0;
}