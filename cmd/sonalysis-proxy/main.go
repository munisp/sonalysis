package main

import (
 "io"
 "log"
 "net/http"
 "os"
 "time"
)

func main() {
 target := os.Getenv("SONALYSIS_API")
 if target == "" { target = "http://sonalysis-api:8080" }
 client := &http.Client{Timeout: 150 * time.Millisecond, Transport: &http.Transport{MaxIdleConns: 4096, MaxIdleConnsPerHost: 4096, IdleConnTimeout: 90 * time.Second}}
 mux := http.NewServeMux()
 mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK); _, _ = w.Write([]byte(`{"status":"ok","proxy":"go"}`)) })
 mux.HandleFunc("/v1/score", func(w http.ResponseWriter, r *http.Request) {
  req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, target+"/v1/score", r.Body); if err != nil { http.Error(w, err.Error(), 400); return }
  req.Header.Set("Content-Type", "application/json"); resp, err := client.Do(req); if err != nil { http.Error(w, "upstream unavailable", 503); return }; defer resp.Body.Close()
  w.WriteHeader(resp.StatusCode); _, _ = io.Copy(w, resp.Body)
 })
 log.Println("sonalysis Go proxy listening on :8090")
 log.Fatal(http.ListenAndServe(":8090", mux))
}
