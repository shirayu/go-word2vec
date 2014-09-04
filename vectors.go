package word2vec

type Vectors []Vector

func (self Vectors) HighestDot(vector Vector) (int, float32) {
	closet_idx := 0
	closet_idx_val := float32(-100.0)
	for idx, vec := range self {
		val := vec.Dot(vector)
		//update
		if val > closet_idx_val {
			closet_idx = idx
			closet_idx_val = val
		}
	}
	return closet_idx, closet_idx_val
}
