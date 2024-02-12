from abc import ABC, abstractmethod


class InterleavingST(ABC):

    @abstractmethod
    def transform(self, *args, **kwargs):
        """Prepare the data for the test."""
        pass

    @abstractmethod
    def calculate_threshold(self, *args, **kwargs):
        """Calculate the thresholds for the test."""
        pass

    @abstractmethod
    def run_test(self, *args, **kwargs):
        """Run the specific test and return the results."""
        pass

    @abstractmethod
    def visualize(self, *args, **kwargs):
        """Visualize the results of the test."""
        pass
