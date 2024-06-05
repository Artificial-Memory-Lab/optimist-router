use std::cmp::Ordering;

// A structure that reports the outcome of the inner product computation for a single document.
#[derive(PartialEq, Clone, Copy, Debug)]
pub struct SearchResult {
    pub docid: u32,
    pub score: f32,
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    #[allow(clippy::non_canonical_partial_ord_impl)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &SearchResult) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// The response of a retrieval algorithm.
#[derive(Default)]
pub struct RetrievalResponse {
    /// The main search results.
    pub results: Vec<SearchResult>,

    /// This is the total number of documents the retrieval algorithm was instructed to
    /// score in order to find the approximate top-k set.
    ///
    /// For example, when using IVF-style search, this reflects the total size of the
    /// top clusters we wish to probe.
    pub expected_number_of_docs_probed: u32,

    /// This is the actual number of documents that were scored to arrive at the top-k set.
    /// This count could be different from `expected_number_of_docs_probed` if the inner MIPS
    /// algorithm operates on an inverted index, for example.
    pub actual_number_of_docs_probed: u32,

    /// Given a query q and a shard P, the following intends to report the average prediction
    /// error defined as follows:
    ///
    ///  | f(q, P) / max_{u \in P} <q, u> - 1 |,
    ///
    /// where f() is the score assigned to shard P for query q by a particular router.
    pub mean_relative_prediction_error: f32,
}
