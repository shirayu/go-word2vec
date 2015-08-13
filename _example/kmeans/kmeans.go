package main

import (
	"bufio"
	"errors"
	"fmt"
	"github.com/jessevdk/go-flags"
	"github.com/shirayu/go-word2vec"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"time"
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
	Log    bool   `long:"log" description:"Enable logging" default:"false"`
}

func operation(opts *cmdOptions) {

	log.Printf("Open the model: %s", opts.Model)
	model, err := getW2VModel(opts.Model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%q\n", err)
		os.Exit(1)
	}

	var targets map[string]struct{} = nil
	if len(opts.Target) != 0 {
		targets = map[string]struct{}{}
		f, err := os.Open(opts.Target)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%q\n", err)
			os.Exit(1)
		}
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			token := scanner.Text()
			targets[token] = struct{}{}
		}
		if err := scanner.Err(); err != nil {
			fmt.Fprintf(os.Stderr, "%q\n", err)
			os.Exit(1)
		}
	}

	vec_size := model.GetVectorSize()
	voc_size := model.GetVocabSize()
	id2word := make([]string, voc_size)
	target_word_num := 0
	for word, i := range model.GetVocab() {
		if targets == nil {
			id2word[i] = word
			target_word_num += 1
			continue
		} else {
			_, ok := targets[word]
			if ok {
				id2word[i] = word
				target_word_num += 1
				continue
			}
		}
		id2word[i] = ""
	}

	log.Printf("Vocabsize: %d\n", voc_size)
	log.Printf("Target words: %d\n", target_word_num)
	log.Printf("Vectorsize: %d\n", vec_size)

	connected_vector := model.GetConnectedVector()
	connected_norm := model.GetNorms()

	centroids := make(word2vec.Vector, vec_size*opts.K)

	iteration := opts.Loop
	for loop := 0; loop < iteration+2; loop++ {
		log.Printf("Loop: %d", loop)
		if loop == 0 {
			log.Printf(" Initialize")
		} else if loop == iteration+1 {
			log.Printf(" Get results")
		}

		new_centroids := make(word2vec.Vector, vec_size*opts.K)
		new_nums := make([]int, opts.K)
		for idx, word := range id2word {
			if len(word) == 0 {
				continue
			}
			myvec := connected_vector[idx*vec_size : (idx+1)*vec_size]
			mynorm := connected_norm[idx]
			myClass := 0

			if loop == 0 { //init
				myClass = rand.Intn(opts.K)
			} else {
				myClass_val := float32(0)
				for n := 0; n < opts.K; n++ {
					cent_vec := centroids[n*vec_size : (n+1)*vec_size]
					val := cent_vec.Dot(myvec) * mynorm
					if val > myClass_val {
						myClass_val = val
						myClass = n
					}
				}
			}

			if loop == iteration+1 {
				fmt.Printf("%d\t%s\n", myClass, word)
			} else {
				cent_vec := new_centroids[myClass*vec_size : (myClass+1)*vec_size]
				cent_vec.Add(mynorm, myvec)
				new_nums[myClass] += 1
			}
		}

		//average
		if loop != iteration+1 {
			for n := 0; n < opts.K; n++ {
				new_cent_vec := new_centroids[n*vec_size : (n+1)*vec_size]
				if new_nums[n] != 0 {
					new_cent_vec.Scal(float32(1) / float32(new_nums[n]))
				}
			}
			centroids = new_centroids
		}
	}

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

	operation(&opts)
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
