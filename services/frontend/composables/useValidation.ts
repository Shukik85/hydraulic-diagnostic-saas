/**
 * Validation Composable
 * Zod-based form validation
 */

import { z } from 'zod';
import type { FormState } from '~/types';

export function useValidation<T extends z.ZodType>(schema: T) {
  type FormData = z.infer<T>;

  const state = ref<FormState<FormData>>({
    values: {} as FormData,
    errors: {},
    touched: {},
    isSubmitting: false,
    isValid: false,
  });

  /**
   * Validate entire form
   */
  function validate(): boolean {
    try {
      schema.parse(state.value.values);
      state.value.errors = {};
      state.value.isValid = true;
      return true;
    } catch (error) {
      if (error instanceof z.ZodError) {
        const errors: Partial<Record<keyof FormData, string>> = {};
        error.errors.forEach((err) => {
          const path = err.path[0] as keyof FormData;
          if (path) {
            errors[path] = err.message;
          }
        });
        state.value.errors = errors;
      }
      state.value.isValid = false;
      return false;
    }
  }

  /**
   * Validate single field
   */
  function validateField(field: keyof FormData): boolean {
    try {
      // Get field schema
      const fieldSchema = schema.shape?.[field as string];
      if (!fieldSchema) return true;

      // Validate field value
      fieldSchema.parse(state.value.values[field]);
      
      // Clear error for this field
      const errors = { ...state.value.errors };
      delete errors[field];
      state.value.errors = errors;
      
      return true;
    } catch (error) {
      if (error instanceof z.ZodError) {
        state.value.errors = {
          ...state.value.errors,
          [field]: error.errors[0]?.message || 'Invalid value',
        };
      }
      return false;
    }
  }

  /**
   * Set field value
   */
  function setValue(field: keyof FormData, value: unknown): void {
    state.value.values = {
      ...state.value.values,
      [field]: value,
    };
  }

  /**
   * Set field as touched
   */
  function setTouched(field: keyof FormData, touched: boolean = true): void {
    state.value.touched = {
      ...state.value.touched,
      [field]: touched,
    };
  }

  /**
   * Handle field blur
   */
  function handleBlur(field: keyof FormData): void {
    setTouched(field, true);
    validateField(field);
  }

  /**
   * Handle field change
   */
  function handleChange(field: keyof FormData, value: unknown): void {
    setValue(field, value);
    if (state.value.touched[field]) {
      validateField(field);
    }
  }

  /**
   * Reset form
   */
  function reset(values?: Partial<FormData>): void {
    state.value = {
      values: (values || {}) as FormData,
      errors: {},
      touched: {},
      isSubmitting: false,
      isValid: false,
    };
  }

  /**
   * Submit handler
   */
  async function handleSubmit(
    onSubmit: (values: FormData) => void | Promise<void>
  ): Promise<void> {
    state.value.isSubmitting = true;

    try {
      // Mark all fields as touched
      const touched: Partial<Record<keyof FormData, boolean>> = {};
      Object.keys(state.value.values).forEach((key) => {
        touched[key as keyof FormData] = true;
      });
      state.value.touched = touched;

      // Validate
      if (!validate()) {
        return;
      }

      // Submit
      await onSubmit(state.value.values);
    } finally {
      state.value.isSubmitting = false;
    }
  }

  return {
    // State
    values: computed(() => state.value.values),
    errors: computed(() => state.value.errors),
    touched: computed(() => state.value.touched),
    isSubmitting: computed(() => state.value.isSubmitting),
    isValid: computed(() => state.value.isValid),

    // Methods
    validate,
    validateField,
    setValue,
    setTouched,
    handleBlur,
    handleChange,
    handleSubmit,
    reset,
  };
}
