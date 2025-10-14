// composables/useNotifications.js
export function useNotifications() {
  const notifications = ref([]);
  let notificationId = 0;

  const pushNotification = (type, message) => {
    const id = ++notificationId;
    notifications.value.push({ id, type, message });

    // Автоочистка через 5 секунд
    setTimeout(() => {
      notifications.value = notifications.value.filter(n => n.id !== id);
    }, 5000);
  };

  const removeNotification = (id) => {
    notifications.value = notifications.value.filter(n => n.id !== id);
  };

  return {
    notifications: readonly(notifications),
    pushNotification,
    removeNotification,
  };
}
