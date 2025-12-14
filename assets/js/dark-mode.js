(function() {
  // Function to set theme
  function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }

  // Function to get theme
  function getTheme() {
    return localStorage.getItem('theme') || 'light';
  }

  // Toggle theme function
  function toggleTheme() {
    const currentTheme = getTheme();
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
  }

  // Add event listener to toggle button when DOM is ready
  document.addEventListener('DOMContentLoaded', function() {
    const toggleButton = document.getElementById('dark-mode-toggle');
    if (toggleButton) {
      toggleButton.addEventListener('click', toggleTheme);
    }
  });
})();

