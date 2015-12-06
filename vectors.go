package word2vec

//Vectors is a slice of vectors
type Vectors []Vector

//HighestDot return the highest dot value and the vector for the given vector
func (vecs Vectors) HighestDot(vector Vector) (int, float32) {
	closetIdx := 0
	closetIdxVal := float32(-100.0)
	for idx, vec := range vecs {
		val := vec.Dot(vector)
		//update
		if val > closetIdxVal {
			closetIdx = idx
			closetIdxVal = val
		}
	}
	return closetIdx, closetIdxVal
}
