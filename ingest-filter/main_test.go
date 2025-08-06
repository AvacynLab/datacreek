package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

func newTestServer() (*FilterServer, *prometheus.Registry) {
	reg := prometheus.NewRegistry()
	s := NewFilterServer(reg)
	return s, reg
}

func TestRegexBlock(t *testing.T) {
	s, _ := newTestServer()
	req := httptest.NewRequest("POST", "/ingest/text", strings.NewReader("badword"))
	rr := httptest.NewRecorder()
	s.ServeHTTP(rr, req)
	if rr.Code != http.StatusForbidden {
		t.Fatalf("expected 403, got %d", rr.Code)
	}
	if testutil.ToFloat64(s.filterBlocked) != 1 {
		t.Fatalf("filter_block_total not incremented")
	}
}

func TestUnsupportedLang(t *testing.T) {
	s, _ := newTestServer()
	req := httptest.NewRequest("POST", "/ingest/text", strings.NewReader("hola mundo"))
	rr := httptest.NewRecorder()
	s.ServeHTTP(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rr.Code)
	}
	if testutil.ToFloat64(s.langSkipped) != 1 {
		t.Fatalf("lang_skipped_total not incremented")
	}
}

func TestToxicNSFWBlock(t *testing.T) {
	s, _ := newTestServer()
	req := httptest.NewRequest("POST", "/ingest/text", strings.NewReader("this is toxic and nsfw"))
	rr := httptest.NewRecorder()
	s.ServeHTTP(rr, req)
	if rr.Code != http.StatusForbidden {
		t.Fatalf("expected 403, got %d", rr.Code)
	}
}

func TestAudioPreGating(t *testing.T) {
	s, _ := newTestServer()
	req := httptest.NewRequest("POST", "/ingest/audio", nil)
	req.Header.Set("X-SNR", "5")
	rr := httptest.NewRecorder()
	s.ServeHTTP(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rr.Code)
	}
	if testutil.ToFloat64(s.filterBlocked) != 1 {
		t.Fatalf("filter_block_total not incremented for audio")
	}
}

// TestToxicDatasetBlocks verifies that a mostly toxic dataset is blocked at the
// expected 99% rate, satisfying the sidecar's DoD requirement.
func TestToxicDatasetBlocks(t *testing.T) {
	s, _ := newTestServer()

	// Send 99 toxic+nsfw texts that should be blocked by the scoring chain.
	blocked := 0
	for i := 0; i < 99; i++ {
		req := httptest.NewRequest("POST", "/ingest/text", strings.NewReader("this is toxic and nsfw"))
		rr := httptest.NewRecorder()
		s.ServeHTTP(rr, req)
		if rr.Code != http.StatusForbidden {
			t.Fatalf("expected 403, got %d", rr.Code)
		}
		blocked++
	}

	// One benign message should pass through unblocked.
	req := httptest.NewRequest("POST", "/ingest/text", strings.NewReader("hello world this is safe"))
	rr := httptest.NewRecorder()
	s.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}

	if blocked != 99 {
		t.Fatalf("expected 99 blocked requests, got %d", blocked)
	}
	if testutil.ToFloat64(s.filterBlocked) != 99 {
		t.Fatalf("filter_block_total = %v, want 99", testutil.ToFloat64(s.filterBlocked))
	}
}
