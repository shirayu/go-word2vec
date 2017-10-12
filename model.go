package word2vec

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
)

//Model is word2vec model
type Model struct {
	vocabSize  int
	vectorSize int
	vocab      map[string]int
	data       Vector //connect all vectors
	norms      Vector //connect all vectors
}

//NewModel returns Model
func NewModel(inf *os.File) (model *Model, err error) {
	fi, err := inf.Stat()
	if err != nil {
		return nil, err
	}
	if fi.IsDir() {
		return nil, errors.New("The path points to a directory")
	}

	model = new(Model)
	reader := bufio.NewReader(inf)
	fmt.Fscanln(reader, &model.vocabSize, &model.vectorSize)

	model.vocab = make(map[string]int)
	model.data = make(Vector, model.vectorSize*model.vocabSize)
	model.norms = make(Vector, model.vocabSize)

	tmp := make(Vector, model.vectorSize)
	var myword string
	for i := 0; i < model.vocabSize; i++ {
		bytes, err := reader.ReadBytes(' ')
		if err != nil {
			return model, err
		}

		myword = string(bytes[0 : len(bytes)-1])
		if myword[0] == '\n' {
			myword = myword[1:]
		}
		model.vocab[myword] = i
		binary.Read(reader, binary.LittleEndian, tmp)

		originalNorm := tmp.Normalize()
		model.norms[i] = originalNorm
		copy(model.data[i*model.vectorSize:(i+1)*model.vectorSize], tmp)
	}

	return model, nil
}

//GetVocab returns the map of vocabulary
func (model *Model) GetVocab() map[string]int {
	return model.vocab
}

//GetVocabSize return the size of vocabulary
func (model *Model) GetVocabSize() int {
	return model.vocabSize
}

//GetConnectedVector return the concatenated vector for all words
func (model *Model) GetConnectedVector() Vector {
	return model.data
}

//GetVectorSize return the number of dimension of each vector
func (model *Model) GetVectorSize() int {
	return model.vectorSize
}

//GetVector returns the normalized vector for the given word and its original norm
func (model *Model) GetVector(word string) (Vector, float32) {
	position, ok := model.vocab[word]
	if ok {
		norm := model.norms[position]
		return model.data[position*model.vectorSize : (position+1)*model.vectorSize], norm
	}
	return nil, 0
}

//GetNormalizedVector returns the normalized vector for the given word
func (model *Model) GetNormalizedVector(word string) Vector {
	position, ok := model.vocab[word]
	if ok {
		return model.data[position*model.vectorSize : (position+1)*model.vectorSize]
	}
	return nil
}

//GetNorms returns the concatenated norms for all vector
func (model *Model) GetNorms() Vector {
	return model.norms
}

//GetNorm returns the original norm for the given word
func (model *Model) GetNorm(word string) float32 {
	position, ok := model.vocab[word]
	if ok {
		return model.norms[position]
	}
	return 0
}

//Similarity returns the similarity between two given words
func (model *Model) Similarity(word1, word2 string) (float32, error) {
	vector1 := model.GetNormalizedVector(word1)
	if vector1 == nil {
		return 0, errors.New(word1 + " not found")
	}
	vector2 := model.GetNormalizedVector(word2)
	if vector2 == nil {
		return 0, errors.New(word2 + " not found")
	}

	return vector1.Dot(vector2), nil
}
