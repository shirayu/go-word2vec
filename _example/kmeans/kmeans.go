package main

import (
	"bufio"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/jessevdk/go-flags"
	"github.com/shirayu/go-word2vec"
)

func getW2VModel(ifname string) (model *word2vec.Model, err error) {
	if ifname == "" {
		return nil, errors.New("Word2Vec file name is not given")
	}
	w2f, err := os.Open(ifname)
	//     defer w2f.Close()
	if err != nil {
		return nil, err
	}
	return word2vec.NewModel(w2f)
}

type cmdOptions struct {
	Help   bool   `short:"h" long:"help" description:"Show this help message"`
	Model  string `short:"m" long:"model" description:"The path to the word2vec model"`
	Target string `long:"target" description:"Enable word limitation"`
	K      int    `short:"k" long:"class" description:"The number of clusters" default:"10"`
	Loop   int    `short:"l" long:"loop" description:"The number of iterations" default:"10"`
	Log    bool   `long:"log" description:"Enable logging"`
}

func getTargets(filename string) (map[string]struct{}, error) {
	var targets map[string]struct{}
	if len(filename) != 0 {
		targets = map[string]struct{}{}
		f, err := os.Open(filename)
		if err != nil {
			return nil, err
		}
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			token := scanner.Text()
			targets[token] = struct{}{}
		}
		if err := scanner.Err(); err != nil {
			return nil, err
		}
	}
	return targets, nil
}

func getID2word(model *word2vec.Model, targets map[string]struct{}) []string {
	if model == nil {
		return nil
	}
	vocSize := model.GetVocabSize()
	id2word := make([]string, vocSize)
	targetWordNum := 0
	for word, i := range model.GetVocab() {
		if targets == nil {
			id2word[i] = word
			targetWordNum++
			continue
		} else {
			_, ok := targets[word]
			if ok {
				id2word[i] = word
				targetWordNum++
				continue
			}
		}
		id2word[i] = ""
	}
	log.Printf("Target words: %d\n", targetWordNum)
	return id2word
}

func operation(opts *cmdOptions) error {

	log.Printf("Open the model: %s", opts.Model)
	model, err := getW2VModel(opts.Model)
	if err != nil {
		return err
	}

	targets, err := getTargets(opts.Target)
	if err != nil {
		return err
	}

	vecSize := model.GetVectorSize()
	vocSize := model.GetVocabSize()

	log.Printf("Vocabsize: %d\n", vocSize)
	log.Printf("Vectorsize: %d\n", vecSize)
	id2word := getID2word(model, targets)

	connectedVector := model.GetConnectedVector()
	connectedNorm := model.GetNorms()

	centroids := make(word2vec.Vector, vecSize*opts.K)

	iteration := opts.Loop
	for loop := 0; loop < iteration+2; loop++ {
		log.Printf("Loop: %d", loop)
		if loop == 0 {
			log.Printf(" Initialize")
		} else if loop == iteration+1 {
			log.Printf(" Get results")
		}

		newCentroids := make(word2vec.Vector, vecSize*opts.K)
		newNums := make([]int, opts.K)
		for idx, word := range id2word {
			if len(word) == 0 {
				continue
			}
			myvec := connectedVector[idx*vecSize : (idx+1)*vecSize]
			mynorm := connectedNorm[idx]
			myClass := 0

			if loop == 0 { //init
				myClass = rand.Intn(opts.K)
			} else {
				myClassVal := float32(0)
				for n := 0; n < opts.K; n++ {
					centVec := centroids[n*vecSize : (n+1)*vecSize]
					val := centVec.Dot(myvec) * mynorm
					if val > myClassVal {
						myClassVal = val
						myClass = n
					}
				}
			}

			if loop == iteration+1 {
				fmt.Printf("%d\t%s\n", myClass, word)
			} else {
				centVec := newCentroids[myClass*vecSize : (myClass+1)*vecSize]
				centVec.Add(mynorm, myvec)
				newNums[myClass]++
			}
		}

		//average
		if loop != iteration+1 {
			for n := 0; n < opts.K; n++ {
				newCentVec := newCentroids[n*vecSize : (n+1)*vecSize]
				if newNums[n] != 0 {
					newCentVec.Scal(float32(1) / float32(newNums[n]))
				}
			}
			centroids = newCentroids
		}
	}
	return nil
}

func main() {
	opts := cmdOptions{}
	optparser := flags.NewParser(&opts, flags.Default)
	optparser.Name = ""
	optparser.Usage = "-i input -o output [OPTIONS]"
	_, err := optparser.Parse()
	if err != nil {
		os.Exit(1)
	}

	//show help
	if len(os.Args) == 1 {
		optparser.WriteHelp(os.Stdout)
		os.Exit(0)
	}
	for _, arg := range os.Args {
		if arg == "-h" {
			os.Exit(0)
		}
	}

	if opts.Log == false {
		log.SetOutput(ioutil.Discard)
	}

	err = operation(&opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%q\n", err)
		os.Exit(1)
	}

}

func init() {
	rand.Seed(time.Now().UnixNano())
}
