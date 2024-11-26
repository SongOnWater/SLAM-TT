using UnityEngine;

public class CameraMovement : MonoBehaviour
{
    public Transform[] points; // Array of points to move the camera between.
    public float moveSpeed = 3.0f; // Speed of camera movement.
    public float rotateSpeed = 0.2f;

    private int currentPointIndex = 0;
    private bool isMoving = false;

    void Update()
    {
        if (Input.GetMouseButtonDown(0) && !isMoving) // Detect left mouse button click.
        {
            // Move to the next point in the array.
            currentPointIndex = (currentPointIndex + 1) % points.Length;
            StartCoroutine(MoveToPoint(points[currentPointIndex]));
        }
    }

    private System.Collections.IEnumerator MoveToPoint(Transform targetPoint)
    {
        isMoving = true;
        Vector3 startPosition = transform.position;
        float elapsedTime = 0f;
        float journeyLength = Vector3.Distance(startPosition, targetPoint.position);

        while (elapsedTime < journeyLength / moveSpeed)
        {
            elapsedTime += Time.deltaTime;
            transform.position = Vector3.Lerp(startPosition, targetPoint.position, elapsedTime * moveSpeed / journeyLength);
            transform.rotation = Quaternion.Slerp(transform.rotation, targetPoint.rotation, elapsedTime * rotateSpeed / journeyLength);
            yield return null;
        }

        // Ensure the camera arrives at the exact target position and rotation.
        transform.position = targetPoint.position;
        transform.rotation = targetPoint.rotation;

        isMoving = false;
    }
}
