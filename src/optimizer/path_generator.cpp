/*
 *  path_generator.cpp
 *
 *  Created on: 12/11/2015
 *      Author: Karol Dzitkowski
 */

#include "compression/default_encoding_factory.hpp"

#include <optimizer/path_generator.hpp>
#include <boost/assign/list_of.hpp>
#include <iostream>

namespace ddj
{

PathList PathGenerator::GeneratePaths()
{
	PathList firstStage = boost::assign::list_of<Path>
			(Path {})
			(Path {EncodingType::floatToInt})
			(Path {EncodingType::delta})
			(Path {EncodingType::scale})
			(Path {EncodingType::delta, EncodingType::floatToInt})
			(Path {EncodingType::scale, EncodingType::floatToInt})
			;

	PathList secondStage = boost::assign::list_of<Path>
			(Path
					{
					EncodingType::dict,
						EncodingType::afl, EncodingType::none,
						EncodingType::afl, EncodingType::none
					}
			)
			(Path {EncodingType::constData, EncodingType::afl, EncodingType::none})
			(Path {EncodingType::rle, EncodingType::afl, EncodingType::none})
			(Path {EncodingType::afl, EncodingType::none})
			(Path
					{
					EncodingType::patch,
						EncodingType::afl, EncodingType::none,
						EncodingType::afl, EncodingType::none
					}
			)
			;

	PathList result;

	for(auto& a : firstStage)
		for(auto& b : secondStage)
		{
			Path temp = a;
			temp.insert(temp.end(), b.begin(), b.end());
			result.push_back(temp);
		}

	return result;
}

void CreateTree(CompressionTree& tree, Path& path, DataType type, int parentId, int& id)
{
	auto encType = path[id];
	auto encFactory = DefaultEncodingFactory().Get(encType, type);
	auto node = boost::make_shared<CompressionNode>(encFactory);
	tree.AddNode(node, parentId);
	if(encType == EncodingType::none) return;

	auto encoding = encFactory->Get();
	type = encoding->GetReturnType(type);
	auto childrenCnt = encoding->GetNumberOfResults();
	for(int i = 1; i <= childrenCnt; i++)
		CreateTree(tree, path, type, node->GetNo(), ++id);
}

CompressionTree PathGenerator::GenerateTree(Path path, DataType type)
{
	CompressionTree tree;
	int id = 0;
	CreateTree(tree, path, type, 0, id);
	return tree;
}


} /* namespace ddj */
