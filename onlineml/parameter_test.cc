#include <iostream>

#include "gtest/gtest.h"

#include "parameter.h"


class TensorTest : public ::testing::Test {
};


TEST_F(TensorTest, Numel) {
//  onlineml::Tensor tensor({2, 3});
  onlineml::Tensor tensor = onlineml::arange(18, {2, 3, 3});
  EXPECT_EQ(18, tensor.numel());
}

TEST_F(TensorTest, Arange) {
//  std::cout << "Arange" << std::endl;
  onlineml::Tensor tensor = onlineml::arange(6, {2, 3});
//  onlineml::Tensor tensor = onlineml::arange(18, {2, 3, 3});
//  EXPECT_EQ(5., tensor[0][1][2].scalar());
//  for (unsigned i = 0; i << tensor.numel(); ++i) {
//    std::cout << tensor.data_ptr()[i] << std::endl;
//  }
  EXPECT_EQ(0., tensor[0][0].scalar());
  EXPECT_EQ(1., tensor[0][1].scalar());
}

TEST_F(TensorTest, Add) {
//  std::cout << "Add" << std::endl;
  onlineml::Tensor tensor = onlineml::arange(18, {2, 3, 3});
  tensor.index_add_({0, 1, 2}, 1.);
  EXPECT_EQ(6., tensor[0][1][2].scalar());
}


TEST_F(TensorTest, SaveLoad) {
  onlineml::Tensor tensor = onlineml::arange(18, {2, 3, 3});
  onlineml::save(tensor, "tensor.om");

  onlineml::Tensor tensor2 = onlineml::zeros({2, 3, 3});
  onlineml::load(tensor2, "tensor.om");

  EXPECT_EQ(tensor[1][1][1].scalar(), tensor2[1][1][1].scalar());
  EXPECT_EQ(tensor[1][2][1].scalar(), tensor2[1][2][1].scalar());
  EXPECT_EQ(tensor[1][2][2].scalar(), tensor2[1][2][2].scalar());
}
