template <is_Tensor Left, is_Tensor_or_Arith Right>
static constexpr void multiply(Left &left, Right &right) {
    imp::multiply(left, right);
}

template <is_Tensor Left, is_Tensor_or_Arith Right>
static constexpr void multiply(Left &&left, Right &right) {
    imp::multiply(left, right);
}

template <is_Tensor Left, is_Tensor_or_Arith Right>
static constexpr void multiply(Left &left, Right &&right) {
    imp::multiply(left, right);
}

template <is_Tensor Left, is_Tensor_or_Arith Right>
static constexpr void multiply(Left &&left, Right &&right) {
    imp::multiply(left, right);
}