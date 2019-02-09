#include "StudentLocalization.h"
#include "IntensityImageStudent.h"
#include "ImageFactory.h"
#include "ImageIO.h"

#include <array>

bool StudentLocalization::stepFindHead(const IntensityImage &image, FeatureMap &features) const {
	return false;
}

bool StudentLocalization::stepFindNoseMouthAndChin(const IntensityImage &image, FeatureMap &features) const {
	return false;
}

bool StudentLocalization::stepFindChinContours(const IntensityImage &image, FeatureMap &features) const {
	return false;
}

bool StudentLocalization::stepFindNoseEndsAndEyes(const IntensityImage &image, FeatureMap &features) const {
	return false;
}

void saveDebug(const IntensityImage &image, const std::string &filename)
{
	RGBImage * out = ImageFactory::newRGBImage(image.getWidth(), image.getHeight());

	for (int x = 0; x < image.getWidth(); x++)
	{
		for (int y = 0; y < image.getHeight(); y++)
		{
			out->setPixel(x, y, image.getPixel(x, y));
		}
	}

	ImageIO::saveRGBImage(*out, ImageIO::getDebugFileName(filename));
	delete out;
}

void pointAround(IntensityImage &img, const Point2D<double> &p)
{
	for (auto x = p.x - 1; x < p.x + 1; x++)
	{
		for (auto y=  p.y - 1; y < p.y + 1; y++)
		{
			img.setPixel(x, y, 127);
		}
	}
}

template<int X, int Y>
using kernel = std::array<std::array<bool, X>, Y>;

/**
 * Dilate a given intensity image with the given kernel.
 */
template<int X, int Y>
void dilate(const IntensityImage &source, IntensityImage &dest, const kernel<X, Y> &k, int maxY)
{
	for (int x = 0; x < dest.getWidth(); x++)
	{
		for (int y = 0; y < maxY; y++)
		{
			bool inMask = false;

			const auto centerX = X / 2;
			const auto centerY = Y / 2;

			for (int kx = 0; kx < X; kx++)
			{
				for (int ky = 0; ky < Y; ky++)
				{
					// Check if this position is in the kernel
					if (!k[kx][ky])
					{
						continue;
					}

					// Combine into the x and y to be checked
					const auto currX = x + kx - centerX;
					const auto currY = y + ky - centerY;

					// Bounds check
					if (currX < 0 || currY < 0 || currX > source.getWidth() || currY > source.getHeight())
					{
						continue;
					}

					// Check
					if (source.getPixel(currX, currY) == 0)
					{
						inMask = true;
						break;
					}
				}

				// Early return
				if (inMask)
				{
					break;
				}
			}

			dest.setPixel(x, y, inMask ? 0 : 255);
		}
	}
}

bool StudentLocalization::stepFindExactEyes(const IntensityImage &image, FeatureMap &features) const {
	auto *target = ImageFactory::newIntensityImage(image.getWidth(), image.getHeight());

	saveDebug(image, "incoming.png");

	//Known head parameters.
	Point2D<double> points[] = {
		features.getFeature(Feature::FEATURE_HEAD_LEFT_NOSE_BOTTOM).getPoints()[0],
		features.getFeature(Feature::FEATURE_HEAD_RIGHT_NOSE_BOTTOM).getPoints()[0],
		features.getFeature(Feature::FEATURE_NOSE_END_LEFT).getPoints()[0],
		features.getFeature(Feature::FEATURE_NOSE_END_RIGHT).getPoints()[0],
		features.getFeature(Feature::FEATURE_NOSE_BOTTOM).getPoints()[0]
	};

	auto noseY = features.getFeature(Feature::FEATURE_NOSE_BOTTOM).getPoints()[0].y;

	auto width = points[1] - points[0];

	// DEBUG
	/*
	for (int i = points[0].getX(); i < points[1].getX(); i++)
	{
		target->setPixel(i, points[0].getY(), 127);
	}

	for (const auto &p : points)
	{
		pointAround(*target, p);
	}

	saveDebug(*target, "points.png");
	*/

	// Dilation
	kernel<3, 3> k = { {
		{0, 1, 0},
		{1, 1, 1},
		{0, 1, 0}
	}};

	// Dilation happens in the part of the image above the nose
	dilate(image, *target, k, noseY);

	saveDebug(*target, "dilate.png");


	delete target;
	

	return true;
}