package main

import (
	"bufio"
	"errors"
	"fmt"
	"github.com/jessevdk/go-flags"
	"github.com/shirayu/go-word2vec"
	"io/ioutil"
	"log"
	"os"
	"runtime"
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

func getWordList(ifname string) ([]string, error) {
	f, err := os.Open(ifname)
	if err != nil {
		return nil, err
	}
	r := bufio.NewReader(f)

	wordlist := []string{}
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		wordlist = append(wordlist, scanner.Text())
	}
	return wordlist, nil
}

type cmdOptions struct {
	Help      bool    `short:"h" long:"help" description:"Show this help message"`
	Input     string  `short:"i" long:"input" description:"The path to the word list"`
	Model     string  `short:"m" long:"model"  description:"The path to the word2vec model file"`
	Threthold float32 `short:"t" long:"threthold" description:"The threthold of similarity to output" default:"0.4"`
	Log       bool    `long:"log" description:"Enable logging" default:"false"`
	Parallel  int     `short:"p" long:"parallel" description:"Parallel number to run" default:"1"`
}

func getSims(wordlist []string, model *word2vec.Model, start int, end int, th float32) {
	for idx := start; idx < end; idx++ {
		x := wordlist[idx]
		for idx2, word := range wordlist {
			if idx2 >= idx {
				break
			}
			simval, _ := model.Similarity(x, word)
			if simval >= th {
				fmt.Printf("%s\t%s\t%f\n", x, word, simval)
			}
		}
	}
}

func main() {
	opts := cmdOptions{}
	optparser := flags.NewParser(&opts, flags.Default)
	optparser.Name = ""
	optparser.Usage = "-i wordlist -m modelfile [OPTIONS]"
	optparser.Parse()

	//show help
	if len(os.Args) == 1 {
		optparser.WriteHelp(os.Stderr)
		os.Exit(1)
	}
	for _, arg := range os.Args {
		if arg == "-h" {
			os.Exit(0)
		}
	}
	runtime.GOMAXPROCS(opts.Parallel)

	if opts.Log == false {
		log.SetOutput(ioutil.Discard)
	}

	log.Printf("Model is %s", opts.Model)
	model, err := getW2VModel(opts.Model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%q\n", err)
		os.Exit(1)
	}

	wordlist, err := getWordList(opts.Input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%q\n", err)
		os.Exit(1)
	}
	log.Printf("Wordlist is %s (size=%d)", opts.Input, len(wordlist))

	finished := make(chan struct{})
	limit := make(chan struct{}, opts.Parallel)
	var empty struct{}
	unit := 1
	count := 0
	for start := 0; start < len(wordlist); start += unit {
		select {
		case limit <- empty:
			end := start + unit
			if end > len(wordlist) {
				end = len(wordlist)
			}
			go func(mystart int, myend int) {
				getSims(wordlist, model, mystart, myend, opts.Threthold)
				<-limit
				finished <- empty
			}(start, end)
			count++

			if count%1000 == 0 {
				log.Printf("%d finished", count)
			}
		}
	}
	log.Printf("%d finished", count)

	for i := 0; i < count; i++ {
		<-finished
	}

}
