#include "StudentLocalization.h"
#include "IntensityImageStudent.h"
#include "ImageFactory.h"
#include "ImageIO.h"

#include <vector>
#include <array>
#include <numeric>
#include <queue>
#include <unordered_set>

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
	RGBImage *out = ImageFactory::newRGBImage(image.getWidth(), image.getHeight());

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

void saveHistogram(const std::vector<int> &values, const std::string &filename)
{
	int selected = 0;

	// std::upper_bound won't work here
	for (int i = 0; i < values.size(); i++)
	{
		if (values[i] > values[selected])
		{
			selected = i;
		}
	}

	RGBImage *out = ImageFactory::newRGBImage(values.size(), values[selected]);

	for (int i = 0; i < values.size(); i++)
	{
		// Fill entire column
		for (int j = 0; j < values[i]; j++)
		{
			out->setPixel(i, j, RGB(255, 255, 255));
		}
	}

	// Fill selected column
	for (int i = 0; i < out->getHeight(); i++)
	{
		out->setPixel(selected, i, RGB(80, 80, 80));
	}

	ImageIO::saveRGBImage(*out, ImageIO::getDebugFileName(filename));
	delete out;
}

namespace std {
	template <>
	struct hash<Point2D<int>>
	{
		typedef Point2D<int>      argument_type;
		typedef std::size_t  result_type;

		result_type operator()(const Point2D<int> & t) const noexcept
		{
			const std::hash<int> hasher;
			const result_type seed = hasher(t.x);

			return seed ^ hasher(t.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}
	};
}

template<typename T, typename U>
Point2D<T> convert_point(const Point2D<U> &p)
{
	return {
		static_cast<T>(p.x),
		static_cast<T>(p.y)
	};
}

template<int X, int Y>
using kernel = std::array<std::array<bool, X>, Y>;

template<typename T>
struct rect
{
	T x;
	T y;
	T width;
	T height;

	rect() : x(0), y(0), width(0), height(0) {}

	rect(T x, T y, T width, T height) 
		: x(x), y(y), width(width), height(height) {}
};

class bfs
{
protected:
	struct bfs_entry
	{
		Point2D<int> loc;
		int distance;

		bfs_entry(const Point2D<int> loc, int distance = -1)
			: loc(loc), distance(distance) {}
	};

	const IntensityImage &source;

	static std::array<Point2D<int>, 8> get_neighbours(const Point2D<int> &loc)
	{
		return { {
			{loc.x - 1, loc.y - 1},
			{ loc.x, loc.y - 1 },
			{ loc.x + 1, loc.y - 1 },
			{ loc.x + 1, loc.y },
			{ loc.x + 1, loc.y + 1 },
			{ loc.x, loc.y + 1 },
			{ loc.x - 1, loc.y + 1 },
			{ loc.x - 1, loc.y },
		}};
	}

public:
	explicit bfs(const IntensityImage &source)
		: source(source) {}

	std::unordered_set<Point2D<int>> expand_area(const Point2D<int> start, const int max_depth = 16) const
	{
		std::unordered_set<Point2D<int>> visited({ start });
		std::unordered_set<Point2D<int>> result({ start });

		std::queue<bfs_entry> q;
		q.emplace(start, 0);

		while (!q.empty())
		{
			const auto u = q.front();
			q.pop();

			if (u.distance + 1 >= max_depth)
			{
				continue;
			}

			const auto points = get_neighbours(u.loc);
			for (const auto &p : points)
			{
				if (visited.find(p) != visited.end())
				{
					continue;
				}

				visited.insert(p);

				if (p.x < 0 || p.y < 0 || p.x > source.getWidth() || p.y > source.getHeight())
				{
					continue;
				}

				if (source.getPixel(p.x, p.y) != 0)
				{
					continue;
				}

				result.insert(p);
				q.emplace(p, u.distance + 1);
			}
		}

		return result;
	}
};

class img_wrapper
{
protected:
	IntensityImage *image;

public:
	img_wrapper(int width, int height)
	{
		image = ImageFactory::newIntensityImage(width, height);
	}

	img_wrapper(const img_wrapper &other)
	{
		image = other.image;
	}

	img_wrapper(img_wrapper &&other) noexcept
	{
		image = other.image;
	}

	~img_wrapper()
	{
		delete image;
	}

	IntensityImage &get()
	{
		return *image;
	}

	IntensityImage const &get() const
	{
		return *image;
	}
};

template<typename Func>
void iter(const rect<int> bounds, Func f)
{
	for (auto x = bounds.x; x < bounds.x + bounds.width; x++)
	{
		for (auto y = bounds.y; y < bounds.y + bounds.height; y++)
		{
			f(x, y);
		}
	}
}

template<int X, int Y>
void erode(const IntensityImage &source, IntensityImage &dest, const rect<int> bounds, const kernel<X, Y> &k)
{
	iter(bounds, [&](const int x, const int y)
	{
		bool inMask = true;

		// Explicit rounding
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
				if (currX < 0 || currY < 0 || currX > source.getWidth() || currY > source.getHeight() - 1)
				{
					continue;
				}

				// Check
				if (source.getPixel(currX, currY) == 255)
				{
					inMask = false;
					break;
				}
			}

			// Early return
			if (!inMask)
			{
				break;
			}
		}

		dest.setPixel(
			x - bounds.x,
			y - bounds.y,
			inMask ? 0 : 255
		);
	});
}

/**
 * Dilate a given Tensity image with the given kernel.
 */
template<int X, int Y>
void dilate(const IntensityImage &source, IntensityImage &dest, const rect<int> bounds, const kernel<X, Y> &k)
{
	iter(bounds, [&](const int x, const int y)
	{
		bool inMask = false;

		// Explicit rounding
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
				if (currX < bounds.x || currY < bounds.x || currX > bounds.width || currY > bounds.height - 1)
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

		dest.setPixel(
			x - bounds.x,
			y - bounds.y,
			inMask ? 0 : 255
		);
	});
}

std::vector<int> score_rows(const IntensityImage &source, const rect<int> bounds)
{
	std::vector<int> scores;

	for (int row = 0; row < bounds.height; row++)
	{
		int score = 0;
		for (int col = 0; col < bounds.width; col++)
		{
			if (source.getPixel(bounds.x + col, bounds.y + row) == 0)
			{
				score += 1;
			}
		}

		scores.push_back(score);
	}

	return scores;
}

std::array<Point2D<int>, 2> get_plane_rect(const std::unordered_set<Point2D<int>> &points)
{
	Point2D<int> xy = { INT32_MAX, INT32_MAX };
	Point2D<int> wh = { 0, 0 };

	for (const auto &point : points)
	{
		if (xy.x > point.x)
		{
			xy.x = point.x;
		}

		if (xy.y > point.y)
		{
			xy.y = point.y;
		}

		if (wh.x < point.x)
		{
			wh.x = point.x;
		}

		if (wh.y < point.y)
		{
			wh.y = point.y;
		}

	}

	return { xy, wh };
}

bool StudentLocalization::stepFindExactEyes(const IntensityImage &image, FeatureMap &features) const {

	saveDebug(image, "incoming.png");

	const int headLeft = features.getFeature(Feature::FEATURE_HEAD_LEFT_NOSE_BOTTOM).getPoints()[0].x;
	const int headRight = features.getFeature(Feature::FEATURE_HEAD_RIGHT_NOSE_BOTTOM).getPoints()[0].x;
	const int headTop = features.getFeature(Feature::FEATURE_HEAD_TOP).getPoints()[0].y;
	const auto nose = features.getFeature(Feature::FEATURE_NOSE_BOTTOM).getPoints()[0];

	const rect<int> bounds{
		headLeft,
		headTop,
		headRight - headLeft,
		static_cast<int>(nose.y)
	};

	const kernel<3, 3> erosion_kernel = { {
		{0, 1, 0},
		{1, 1, 1},
		{0, 1, 0}
	} };

	const kernel<3, 3> dilation_kernel = { {
		{0, 1, 0},
		{1, 1, 1},
		{0, 1, 0}
	}};

	auto dilated = img_wrapper(bounds.width, bounds.height);
	auto eroded = img_wrapper(bounds.width, bounds.height);

	// Dilation happens in the part of the image above the nose
	erode(
		image,
		eroded.get(),
		bounds, 
		erosion_kernel
	);

	saveDebug(eroded.get(), "eroded.png");

	dilate(
		eroded.get(),
		dilated.get(),
		{ 
			0,
			0,
			bounds.width,
			bounds.height 
		},
		dilation_kernel
	);

	const auto &dilatedPtr = dilated.get();
	const int eyeBrowOffset = dilatedPtr.getHeight() / 10;
	const int scoreOffsetY = dilatedPtr.getHeight() / 2 + eyeBrowOffset;

	const auto scores = score_rows(
		dilated.get(), {
		0,
		scoreOffsetY,
		dilatedPtr.getWidth(),
		dilatedPtr.getHeight() - scoreOffsetY - headTop
	});

	saveHistogram(scores, "histogram.png");

	int selected = 0;

	// std::upper_bound won't work here
	for (int i = 0; i < scores.size(); i++)
	{
		if (scores[i] > scores[selected])
		{
			selected = i;
		}
	}

	selected += scoreOffsetY;

	/**
	 * Find the left eye, searching from the left on the
	 * selected row.
	 */
	Point2D<int> start(0, selected);
	for (int col = 0; col < bounds.width; col++)
	{
		if (!dilated.get().getPixel(col, selected))
		{
			start.x = col;
			break;
		}
	}

	/**
	 * Find the right eye, searching from the right on the 
	 * selected row.
	 */
	const auto width = dilated.get().getWidth();
	Point2D<int> end(0, selected);
	for (int col = width; col != 0; col--)
	{
		if (!dilated.get().getPixel(col, selected))
		{
			end.x = col;
			break;
		}
	}

	// Expand the eye planes
	bfs b(dilated.get());

	const auto left_eye = b.expand_area(start, 24);
	const auto right_eye = b.expand_area(end, 24);

	for (const auto p : left_eye)
	{
		dilated.get().setPixel(p.x, p.y, 127);
	}

	for (const auto p : right_eye)
	{
		dilated.get().setPixel(p.x, p.y, 127);
	}

	saveDebug(dilated.get(), "dilate.png");

	const auto leftEyePoints = get_plane_rect(left_eye);
	const auto rightEyePoints = get_plane_rect(right_eye);

	Feature leftEye(Feature::FEATURE_EYE_LEFT_RECT);
	for (auto p : leftEyePoints)
	{
		//p.x += headLeft;

		leftEye.addPoint(
			convert_point<double>(p)
		);
	}


	Feature rightEye(Feature::FEATURE_EYE_RIGHT_RECT);
	for (auto p : rightEyePoints)
	{
		rightEye.addPoint(
			convert_point<double>(p)
		);
	}

	features.putFeature(leftEye);
	features.putFeature(rightEye);

	return true;
}