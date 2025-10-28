export default defineNuxtRouteMiddleware(to => {
  const demo = useState('demoMode');
  if (demo?.value === true) return;
});
