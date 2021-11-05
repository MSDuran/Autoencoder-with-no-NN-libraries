#include <iostream>
#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdlib>
#include <random>
#include <boost/filesystem.hpp>

#define ITERATION_LENGTH 1'000
#define IMAGES_COUNT 70'000
#define WIDTH 28
#define HEIGHT 28

float generate_random_float() {
	double value;
	if (value = ((double)std::rand() / (RAND_MAX)); value < 0.4)
		value += 0.4;
	return value;
}

int main()
{
	std::vector<cv::Mat> images;
	images.reserve(IMAGES_COUNT);

	std::string path_to_images;
	std::cout << "Path to images: ";
	std::getline(std::cin, path_to_images);
	if (path_to_images.empty())
		path_to_images = "C:/Users/90543/source/repos/mnist_autoencoder/mnist_png/testing/0";
	size_t image_number = 0;
	for (auto const& dir_entry : boost::filesystem::recursive_directory_iterator{ path_to_images })
	{
		//boost::filesystem::path path{ dir_entry };
		//if (boost::filesystem::is_regular_file(dir_entry))
		if (dir_entry.path().extension() == ".png")
		{
			images.push_back(cv::imread(dir_entry.path().generic_string(), cv::ImreadModes::IMREAD_GRAYSCALE));
			//std::cout << dir_entry << '\n';
			if (image_number++; image_number % 1'000 == 0)
				std::cout << image_number << '\n';
		}
	}
	std::cout << "Total images:" << image_number << "\n";
	std::vector<std::vector<float>> flattened_images;
	for (size_t current_image = 0; current_image < image_number; current_image++) // normalize each input
	{
		flattened_images.push_back(std::vector<float>{});
		images[current_image].convertTo(images[current_image], CV_32F);
		cv::normalize(images[current_image], images[current_image], 0, 255, cv::NORM_MINMAX, -1);
		for (size_t i = 0; i < WIDTH; i++)
		{
			for (size_t j = 0; j < HEIGHT; j++)
			{
				//std::cout << images[current_image].at<float>(i, j) / 255 << " ";
				images[current_image].at<float>(i, j) = images[current_image].at<float>(i, j) / 255.0;
				flattened_images[current_image].push_back(images[current_image].at<float>(i, j));
			}
		}
		//cv::normalize(images[current_image], images[current_image], 0, 255, cv::NORM_MINMAX, -1);
		//cv::Mat out_img;
		//cv::resize(images[current_image], out_img, cv::Size(28, 28));
		//cv::imwrite("C:/Users/90543/source/repos/mnist_autoencoder/" + std::to_string(current_image) + ".png", out_img);
		//if (current_image == 5'000)
		break;
	}
	std::vector<float> W_0(WIDTH * HEIGHT * 512);
	std::vector<float> B_0(512);
	std::vector<float> layer_1(512);
	std::generate_n(W_0.begin(), WIDTH * HEIGHT * 512, generate_random_float);
	std::generate_n(B_0.begin(), 512, std::mt19937{ std::random_device{}() });
	for (size_t iteration_count = 0; iteration_count < ITERATION_LENGTH; iteration_count++)
	{
		for (size_t current_image = 0; current_image < image_number; current_image++) // train
		{
			for (size_t i = 0; i < WIDTH * HEIGHT; i++)
			{
				for (size_t j = 0; j < 512; j++)
				{
					for (size_t k = 0; k < 512; k+=512)
					{
						for (size_t l = 0; l < k; l++)
						{
							layer_1.push_back(flattened_images[current_image].at(i) * W_0[l] + B_0[j]);
						}
					}
				}
			}
		}
	}
	for (size_t current_image = 0; current_image < image_number; current_image++) // save outputs
	{
		cv::normalize(images[current_image], images[current_image], 0, 255, cv::NORM_MINMAX, -1);
		cv::Mat out_img;
		cv::resize(images[current_image], out_img, cv::Size(28, 28));
		cv::imwrite("C:/Users/90543/source/repos/mnist_autoencoder/" + std::to_string(current_image) + ".png", out_img);
		break;
	}
	//784(input) ---> 512 --> 20(latent space) --> 512 --> 784(output) 
	return 0;
}
