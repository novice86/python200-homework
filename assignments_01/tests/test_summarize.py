import pytest

from weatherkit import DailyAggregator, DailySummary, HourlyReading

@pytest.fixture
def sample_readings():
    """
    Shared input data containing 5 readings over 2 days.
    Day 1 has 3 readings.
    Day 2 has 2 readings.
    """
    return [
        # Day 1: 3 readings (Min: 10.0, Max: 15.0, Precip Total: 0.3)
        HourlyReading(timestamp="2026-10-01T08:00", temperature_c=10.0, precipitation_mm=0.1),
        HourlyReading(timestamp="2026-10-01T12:00", temperature_c=15.0, precipitation_mm=0.2),
        HourlyReading(timestamp="2026-10-01T16:00", temperature_c=12.0, precipitation_mm=0.0),
        
        # Day 2: 2 readings (Min: 8.0, Max: 9.0, Precip Total: 1.0)
        HourlyReading(timestamp="2026-10-02T09:00", temperature_c=8.0, precipitation_mm=0.5),
        HourlyReading(timestamp="2026-10-02T10:00", temperature_c=9.0, precipitation_mm=0.5),
    ]

def test_grouping_produces_summaries_for_distinct_days(sample_readings):
    """Ensures a list spanning two dates produces exactly two summaries"""
    aggregator = DailyAggregator(min_hours=2)
    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 2
    assert summaries[0].date == "2026-10-01"
    assert summaries[1].date == "2026-10-02"

# if we modify summarize function by swapping min and max values than 
# this test will fail because the expected maximum value doesn't match the 
# maximum in the DailySummary object
def test_temp_max_and_temp_min_are_correct(sample_readings):
    """Ensures that highest and lowest temperatures are extracted correctly"""
    aggregator = DailyAggregator(min_hours=2)
    summaries = aggregator.summarize(sample_readings)

    day_1_summary = summaries[0]

    assert day_1_summary.temp_max == 15.0
    assert day_1_summary.temp_min == 10.0


def test_precipitation_sum_adds_up_correctly(sample_readings):
    """Verifies precipitation addition using pytest.approx to avoid floating point math errors"""
    aggregator = DailyAggregator(min_hours=2)
    summaries = aggregator.summarize(sample_readings)

    assert summaries[0].precipitation_sum == pytest.approx(0.3)
    assert summaries[1].precipitation_sum == pytest.approx(1.0)


def test_days_below_min_hours_are_dropped_and_logged(sample_readings):
    """Ensures a day with too few readings is dropped and recorded in incomplete_days()"""
    aggregator = DailyAggregator(min_hours=3)
    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 1
    assert summaries[0].date == "2026-10-01"

    assert "2026-10-02" in aggregator.incomplete_days(sample_readings)


def test_lowering_min_hours_keeps_otherwise_dropped_day(sample_readings):
    """Ensures the min_hours parameter dictates the dropping behavior"""

    aggregator = DailyAggregator(min_hours=2)
    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 2
    assert len(aggregator.incomplete_days(sample_readings)) == 0
    assert "2026-10-02" not in aggregator.incomplete_days(sample_readings)


