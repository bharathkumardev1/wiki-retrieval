package passages

import (
	"strings"
	"testing"
)

func para(n int) string { return strings.TrimSpace(strings.Repeat("word ", n-1)) + " end." }

func TestIDsAreStableAndShortChunksDropped(t *testing.T) {
	paras := []string{para(100), para(100), para(10)}
	a := Split(7, "T", paras)
	b := Split(7, "T", paras)
	if len(a) != 2 {
		t.Fatalf("want 2 passages, got %d", len(a))
	}
	if a[0].ID != "7-0" || a[1].ID != "7-1" || a[0].Content != b[0].Content {
		t.Fatalf("ids not stable: %+v", a)
	}
	if a[1].Words != 110 {
		t.Fatalf("trailing short paragraph should merge into the last chunk, got %d words", a[1].Words)
	}
}

func TestLongParagraphIsCut(t *testing.T) {
	got := Split(1, "T", []string{strings.Repeat(para(50)+" ", 8)})
	if len(got) < 2 {
		t.Fatalf("expected a long paragraph to be cut, got %d", len(got))
	}
	for _, p := range got {
		if p.Words > TargetWords+50 {
			t.Fatalf("passage too long: %d", p.Words)
		}
	}
}
