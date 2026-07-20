use crate::Valor;
use std::collections::HashMap;
use std::io::Read;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Replicatio {
    status: i64,
    corpus: String,
    corpus_octeti: Vec<u8>,
    capita: HashMap<String, String>,
}

impl Replicatio {
    #[must_use]
    pub fn new(status: i64, corpus_octeti: Vec<u8>, capita: HashMap<String, String>) -> Self {
        let corpus = String::from_utf8_lossy(&corpus_octeti).into_owned();
        Self {
            status,
            corpus,
            corpus_octeti,
            capita: normalize_headers(capita),
        }
    }

    #[must_use]
    pub fn status(&self) -> i64 {
        self.status
    }

    #[must_use]
    pub fn corpus(&self) -> String {
        self.corpus.clone()
    }

    #[must_use]
    pub fn corpus_octeti(&self) -> Vec<u8> {
        self.corpus_octeti.clone()
    }

    pub fn corpus_json(&self) -> Valor {
        crate::Json::parse(&self.corpus).map_or(Valor::Nihil, Valor::from)
    }

    #[must_use]
    pub fn capita(&self) -> HashMap<String, String> {
        self.capita.clone()
    }

    #[must_use]
    pub fn caput(&self, nomen: String) -> Option<String> {
        self.capita.get(&nomen.to_ascii_lowercase()).cloned()
    }

    #[must_use]
    pub fn bene(&self) -> bool {
        (200..=299).contains(&self.status)
    }
}

pub async fn petet(url: String) -> Replicatio {
    rogabit("GET".to_owned(), url, HashMap::new(), String::new()).await
}

pub async fn mittet(url: String, corpus: String) -> Replicatio {
    rogabit("POST".to_owned(), url, HashMap::new(), corpus).await
}

pub async fn ponet(url: String, corpus: String) -> Replicatio {
    rogabit("PUT".to_owned(), url, HashMap::new(), corpus).await
}

pub async fn delet(url: String) -> Replicatio {
    rogabit("DELETE".to_owned(), url, HashMap::new(), String::new()).await
}

pub async fn mutabit(url: String, corpus: String) -> Replicatio {
    rogabit("PATCH".to_owned(), url, HashMap::new(), corpus).await
}

pub async fn rogabit(
    modus: String,
    url: String,
    capita: HashMap<String, String>,
    corpus: String,
) -> Replicatio {
    match http_request(&modus, &url, &capita, corpus.as_bytes()) {
        Ok(response) => response,
        Err(error) => Replicatio::new(
            599,
            error.into_bytes(),
            HashMap::from([("x-faber-error".to_owned(), "http-client".to_owned())]),
        ),
    }
}

fn http_request(
    method: &str,
    url: &str,
    headers: &HashMap<String, String>,
    body: &[u8],
) -> Result<Replicatio, String> {
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(30))
        .build();
    let mut request = agent.request(method, url);
    for (name, value) in headers {
        request = request.set(name, value);
    }

    let result = if body.is_empty() {
        request.call()
    } else {
        request.send_bytes(body)
    };
    match result {
        Ok(response) => response_to_replicatio(response),
        Err(ureq::Error::Status(_, response)) => response_to_replicatio(response),
        Err(error) => Err(format!("http request failed: {error}")),
    }
}

fn response_to_replicatio(response: ureq::Response) -> Result<Replicatio, String> {
    let status = i64::from(response.status());
    let headers = response
        .headers_names()
        .into_iter()
        .filter_map(|name| response.header(&name).map(|value| (name, value.to_owned())))
        .collect::<HashMap<_, _>>();
    let mut body = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut body)
        .map_err(|error| format!("http body read failed: {error}"))?;
    Ok(Replicatio::new(status, body, headers))
}

fn normalize_headers(headers: HashMap<String, String>) -> HashMap<String, String> {
    headers
        .into_iter()
        .map(|(name, value)| (name.to_ascii_lowercase(), value))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_response_carrier() {
        let response = Replicatio::new(
            201,
            b"{\"ok\":true}".to_vec(),
            HashMap::from([("X-Faber-Test".to_owned(), "yes".to_owned())]),
        );

        assert_eq!(response.status(), 201);
        assert_eq!(
            response.caput("x-faber-test".to_owned()),
            Some("yes".to_owned())
        );
        assert!(matches!(response.corpus_json(), Valor::Tabula(_)));
        assert!(response.bene());
    }
}
