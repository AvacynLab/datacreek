package main

import (
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"

	"github.com/abadojack/whatlanggo"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// FilterServer implements the pre-ingestion safety and gating pipeline.
type FilterServer struct {
	mux           *http.ServeMux
	filterBlocked prometheus.Counter
	langSkipped   prometheus.Counter
}

// NewFilterServer builds a server with its own Prometheus registry.
func NewFilterServer(reg *prometheus.Registry) *FilterServer {
	fb := prometheus.NewCounter(prometheus.CounterOpts{
		Name: "filter_block_total",
		Help: "Number of requests blocked by safety filter",
	})
	ls := prometheus.NewCounter(prometheus.CounterOpts{
		Name: "lang_skipped_total",
		Help: "Number of requests skipped due to unsupported language",
	})
	reg.MustRegister(fb, ls)

	s := &FilterServer{mux: http.NewServeMux(), filterBlocked: fb, langSkipped: ls}
	s.mux.HandleFunc("/ingest/text", s.handleText)
	s.mux.HandleFunc("/ingest/audio", s.handleAudio)
	s.mux.HandleFunc("/ingest/img", s.handleImg)
	s.mux.Handle("/metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{}))
	return s
}

func (s *FilterServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
}

var blacklist = regexp.MustCompile(`(?i)badword`)

func detectLang(text string) string {
	info := whatlanggo.Detect(text)
	return info.Lang.Iso6391()
}

func scoreToxic(text string) float64 {
	if strings.Contains(strings.ToLower(text), "toxic") {
		return 0.9
	}
	return 0.1
}

func scoreNSFW(text string) float64 {
	if strings.Contains(strings.ToLower(text), "nsfw") {
		return 0.9
	}
	return 0.1
}

// handleText applies blacklist, language detection and toxicity scores.
func (s *FilterServer) handleText(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	text := string(body)
	if blacklist.MatchString(text) {
		s.filterBlocked.Inc()
		http.Error(w, "blocked", http.StatusForbidden)
		return
	}
	lang := detectLang(text)
	if lang != "en" && lang != "fr" {
		s.langSkipped.Inc()
		http.Error(w, "unsupported language", http.StatusBadRequest)
		return
	}
	tox := scoreToxic(text)
	nsfw := scoreNSFW(text)
	score := 0.5 * (tox + nsfw)
	if score > 0.7 {
		s.filterBlocked.Inc()
		http.Error(w, "blocked", http.StatusForbidden)
		return
	}
	w.WriteHeader(http.StatusOK)
}

// handleAudio checks SNR before forwarding audio.
func (s *FilterServer) handleAudio(w http.ResponseWriter, r *http.Request) {
	snr, _ := strconv.ParseFloat(r.Header.Get("X-SNR"), 64)
	if snr < 10 {
		s.filterBlocked.Inc()
		http.Error(w, "low snr", http.StatusBadRequest)
		return
	}
	w.WriteHeader(http.StatusOK)
}

// handleImg rejects blurry images using a blur metric.
func (s *FilterServer) handleImg(w http.ResponseWriter, r *http.Request) {
	blur, _ := strconv.ParseFloat(r.Header.Get("X-Blur"), 64)
	if blur > 100 {
		s.filterBlocked.Inc()
		http.Error(w, "blurry", http.StatusBadRequest)
		return
	}
	w.WriteHeader(http.StatusOK)
}

func main() {
	reg := prometheus.NewRegistry()
	s := NewFilterServer(reg)
	http.ListenAndServe(":8080", s)
}
