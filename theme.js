const storageKey = 'moespeedcalc-theme';
const root = document.documentElement;
const toggle = document.getElementById('theme-toggle');

function getPreferredTheme() {
  const savedTheme = window.localStorage.getItem(storageKey);
  if (savedTheme === 'light' || savedTheme === 'dark') {
    return savedTheme;
  }

  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function applyTheme(theme) {
  root.dataset.theme = theme;

  if (!toggle) {
    return;
  }

  const isDark = theme === 'dark';
  toggle.textContent = isDark ? 'Light mode' : 'Dark mode';
  toggle.setAttribute('aria-pressed', String(isDark));
}

applyTheme(getPreferredTheme());

if (toggle) {
  toggle.addEventListener('click', () => {
    const nextTheme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    window.localStorage.setItem(storageKey, nextTheme);
    applyTheme(nextTheme);
  });
}
