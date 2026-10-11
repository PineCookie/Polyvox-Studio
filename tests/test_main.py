import logging

from polyvox.__main__ import QuietReads


def access(method: str, status: int) -> logging.LogRecord:
    return logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "",
        0,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:1", method, "/x", "1.1", status),
        None,
    )


def test_successful_reads_are_left_out_of_the_access_log():
    quiet = QuietReads()
    assert not quiet.filter(access("GET", 200))
    assert not quiet.filter(access("GET", 304))
    assert quiet.filter(access("GET", 404))
    assert quiet.filter(access("POST", 200))
    assert quiet.filter(logging.LogRecord("uvicorn.access", logging.INFO, "", 0, "other", None, None))
